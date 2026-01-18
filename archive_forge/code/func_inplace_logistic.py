import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy
def inplace_logistic(X):
    """Compute the logistic function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    logistic_sigmoid(X, out=X)