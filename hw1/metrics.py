import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    accuracy = np.sum(y_pred == y_true) / len(y_pred)
    
    TP = np.sum(np.logical_and(y_pred, y_true))
    FP = np.sum(np.greater(y_pred, y_true))
    FN = np.sum(np.greater(y_true, y_pred))
    
    precision = TP / (FP + TP)
    
    recall = TP / (TP + FN)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accuracy = np.sum(y_pred == y_true) / len(y_pred)
    
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    y_true = y_true.tolist()
   
    mean_true = np.mean(y_true)
    ss_t = 0
    ss_r = 0
    
    for i in range(len(y_true)):
        ss_t += (y_true[i] - mean_true) ** 2
        ss_r += (y_pred[i] - y_true[i]) ** 2
        
    r2 = 1 - (ss_r/ss_t)
    
    return r2

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    MSE = np.square(np.subtract(y_true, y_pred)).mean()
    
    return MSE


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    
    y_true = y_true.tolist()
    
    n = len(y_true)
    
    summ = 0
    
    for i in range(n):
        summ += abs(y_true[i]-y_pred[i])
        
    MAE = summ/n
    
    return MAE