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
def mean_difference(X, Y):
    """Calculate the mean difference in genes between two datasets.

    In the case where the data has been log normalized,
    this is equivalent to fold change.

    Parameters
    ----------
    X : array-like, shape=[n_cells, n_genes]
    Y : array-like, shape=[m_cells, n_genes]

    Returns
    -------
    difference : list-like, shape=[n_genes]
    """
    X, Y = _preprocess_test_matrices(X, Y)
    X = utils.toarray(X.mean(axis=0)).flatten()
    Y = utils.toarray(Y.mean(axis=0)).flatten()
    return X - Y