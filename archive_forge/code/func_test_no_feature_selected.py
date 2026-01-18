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
def test_no_feature_selected():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    y = rng.randint(0, 4, size=40)
    strict_selectors = [SelectFwe(alpha=0.01).fit(X, y), SelectFdr(alpha=0.01).fit(X, y), SelectFpr(alpha=0.01).fit(X, y), SelectPercentile(percentile=0).fit(X, y), SelectKBest(k=0).fit(X, y)]
    for selector in strict_selectors:
        assert_array_equal(selector.get_support(), np.zeros(10))
        with pytest.warns(UserWarning, match='No features were selected'):
            X_selected = selector.transform(X)
        assert X_selected.shape == (40, 0)