import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def single_fdr(alpha, n_informative, random_state):
    X, y = make_regression(n_samples=150, n_features=20, n_informative=n_informative, shuffle=False, random_state=random_state, noise=10)
    with warnings.catch_warnings(record=True):
        univariate_filter = SelectFdr(f_regression, alpha=alpha)
        X_r = univariate_filter.fit(X, y).transform(X)
        X_r2 = GenericUnivariateSelect(f_regression, mode='fdr', param=alpha).fit(X, y).transform(X)
    assert_array_equal(X_r, X_r2)
    support = univariate_filter.get_support()
    num_false_positives = np.sum(support[n_informative:] == 1)
    num_true_positives = np.sum(support[:n_informative] == 1)
    if num_false_positives == 0:
        return 0.0
    false_discovery_rate = num_false_positives / (num_true_positives + num_false_positives)
    return false_discovery_rate