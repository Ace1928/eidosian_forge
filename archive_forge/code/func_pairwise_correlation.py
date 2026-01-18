from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def pairwise_correlation(X, Y, ignore_nan=False):
    """Compute pairwise Pearson correlation between columns of two matrices.

    From https://stackoverflow.com/a/33651442/3996580

    Parameters
    ----------
    X : array-like, shape=[n_samples, m_features]
        Input data
    Y : array-like, shape=[n_samples, p_features]
        Input data
    ignore_nan : bool, optional (default: False)
        If True, ignore NaNs, computing correlation over remaining values

    Returns
    -------
    cor : np.ndarray, shape=[m_features, p_features]
    """
    N = X.shape[0]
    assert Y.shape[0] == N
    assert len(X.shape) <= 2
    assert len(Y.shape) <= 2
    X = utils.to_array_or_spmatrix(X).reshape(N, -1)
    Y = utils.to_array_or_spmatrix(Y).reshape(N, -1)
    if sparse.issparse(X) and (not sparse.issparse(Y)):
        Y = sparse.csr_matrix(Y)
    if sparse.issparse(Y) and (not sparse.issparse(X)):
        X = sparse.csr_matrix(X)
    X_colsums = utils.matrix_sum(X, axis=0, ignore_nan=ignore_nan)
    Y_colsums = utils.matrix_sum(Y, axis=0, ignore_nan=ignore_nan)
    X_sq_colsums = utils.matrix_sum(utils.matrix_transform(X, np.power, 2), axis=0, ignore_nan=ignore_nan)
    Y_sq_colsums = utils.matrix_sum(utils.matrix_transform(Y, np.power, 2), axis=0, ignore_nan=ignore_nan)
    var_x = N * X_sq_colsums - X_colsums ** 2
    var_y = N * Y_sq_colsums - Y_colsums ** 2
    if ignore_nan:
        X = utils.fillna(X, 0)
        Y = utils.fillna(Y, 0)
    N_times_sum_xy = utils.toarray(N * Y.T.dot(X))
    sum_x_times_sum_y = X_colsums * Y_colsums[:, None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in true_divide', category=RuntimeWarning)
        cor = (N_times_sum_xy - sum_x_times_sum_y) / np.sqrt(var_x * var_y[:, None])
    return cor.T