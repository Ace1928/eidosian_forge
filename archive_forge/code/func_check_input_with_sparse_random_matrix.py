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
def check_input_with_sparse_random_matrix(random_matrix):
    n_components, n_features = (5, 10)
    for density in [-1.0, 0.0, 1.1]:
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features, density=density)