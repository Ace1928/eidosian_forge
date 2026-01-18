import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_regression():
    X, y, c = make_regression(n_samples=100, n_features=10, n_informative=3, effective_rank=5, coef=True, bias=0.0, noise=1.0, random_state=0)
    assert X.shape == (100, 10), 'X shape mismatch'
    assert y.shape == (100,), 'y shape mismatch'
    assert c.shape == (10,), 'coef shape mismatch'
    assert sum(c != 0.0) == 3, 'Unexpected number of informative features'
    assert_almost_equal(np.std(y - np.dot(X, c)), 1.0, decimal=1)
    X, y = make_regression(n_samples=100, n_features=1)
    assert X.shape == (100, 1)