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
@pytest.mark.parametrize('random_matrix', all_sparse_random_matrix)
def test_basic_property_of_sparse_random_matrix(random_matrix):
    check_input_with_sparse_random_matrix(random_matrix)
    random_matrix_dense = functools.partial(random_matrix, density=1.0)
    check_zero_mean_and_unit_norm(random_matrix_dense)