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
def test_invalid_k():
    X = [[0, 1, 0], [0, -1, -1], [0, 0.5, 0.5]]
    y = [1, 0, 1]
    msg = 'k=4 is greater than n_features=3. All the features will be returned.'
    with pytest.warns(UserWarning, match=msg):
        SelectKBest(k=4).fit(X, y)
    with pytest.warns(UserWarning, match=msg):
        GenericUnivariateSelect(mode='k_best', param=4).fit(X, y)