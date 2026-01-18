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
def check_zero_mean_and_unit_norm(random_matrix):
    A = densify(random_matrix(10000, 1, random_state=0))
    assert_array_almost_equal(0, np.mean(A), 3)
    assert_array_almost_equal(1.0, np.linalg.norm(A), 1)