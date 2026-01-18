import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_imputation_mean_median(csc_container):
    rng = np.random.RandomState(0)
    dim = 10
    dec = 10
    shape = (dim * dim, dim + dec)
    zeros = np.zeros(shape[0])
    values = np.arange(1, shape[0] + 1)
    values[4::2] = -values[4::2]
    tests = [('mean', np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))), ('median', np.nan, lambda z, v, p: safe_median(np.hstack((z, v))))]
    for strategy, test_missing_values, true_value_fun in tests:
        X = np.empty(shape)
        X_true = np.empty(shape)
        true_statistics = np.empty(shape[1])
        for j in range(shape[1]):
            nb_zeros = (j - dec + 1 > 0) * (j - dec + 1) * (j - dec + 1)
            nb_missing_values = max(shape[0] + dec * dec - (j + dec) * (j + dec), 0)
            nb_values = shape[0] - nb_zeros - nb_missing_values
            z = zeros[:nb_zeros]
            p = np.repeat(test_missing_values, nb_missing_values)
            v = values[rng.permutation(len(values))[:nb_values]]
            true_statistics[j] = true_value_fun(z, v, p)
            X[:, j] = np.hstack((v, z, p))
            if 0 == test_missing_values:
                X_true[:, j] = np.hstack((v, np.repeat(true_statistics[j], nb_missing_values + nb_zeros)))
            else:
                X_true[:, j] = np.hstack((v, z, np.repeat(true_statistics[j], nb_missing_values)))
            np.random.RandomState(j).shuffle(X[:, j])
            np.random.RandomState(j).shuffle(X_true[:, j])
        if strategy == 'median':
            cols_to_keep = ~np.isnan(X_true).any(axis=0)
        else:
            cols_to_keep = ~np.isnan(X_true).all(axis=0)
        X_true = X_true[:, cols_to_keep]
        _check_statistics(X, X_true, strategy, true_statistics, test_missing_values, csc_container)