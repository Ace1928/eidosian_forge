import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy
def inplace_softmax(X):
    """Compute the K-way softmax function inplace.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]