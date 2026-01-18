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
def rank_sum_statistic(X, Y):
    """Calculate the Wilcoxon rank-sum (aka Mann-Whitney U) statistic.

    Parameters
    ----------
    X : array-like, shape=[n_cells, n_genes]
    Y : array-like, shape=[m_cells, n_genes]

    Returns
    -------
    rank_sum_statistic : list-like, shape=[n_genes]
    """
    X, Y = _preprocess_test_matrices(X, Y)
    data, labels = utils.combine_batches([X, Y], ['x', 'y'])
    X_rank_sum = _ranksum(data, labels == 'x', axis=0)
    X_u_statistic = X_rank_sum - X.shape[0] * (X.shape[0] + 1) / 2
    Y_u_statistic = X.shape[0] * Y.shape[0] - X_u_statistic
    return np.minimum(X_u_statistic, Y_u_statistic)