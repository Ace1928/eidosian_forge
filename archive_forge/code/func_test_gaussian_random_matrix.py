import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
def test_gaussian_random_matrix():
    n_components = 100
    n_features = 1000
    A = _gaussian_random_matrix(n_components, n_features, random_state=0)
    assert_array_almost_equal(0.0, np.mean(A), 2)
    assert_array_almost_equal(np.var(A, ddof=1), 1 / n_components, 1)