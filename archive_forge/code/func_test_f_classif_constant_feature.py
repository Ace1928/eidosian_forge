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
def test_f_classif_constant_feature():
    X, y = make_classification(n_samples=10, n_features=5)
    X[:, 0] = 2.0
    with pytest.warns(UserWarning):
        f_classif(X, y)