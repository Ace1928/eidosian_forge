import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy
def inplace_relu(X):
    """Compute the rectified linear unit function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    np.maximum(X, 0, out=X)