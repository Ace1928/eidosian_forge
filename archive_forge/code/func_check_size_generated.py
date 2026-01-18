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
def check_size_generated(random_matrix):
    inputs = [(1, 5), (5, 1), (5, 5), (1, 1)]
    for n_components, n_features in inputs:
        assert random_matrix(n_components, n_features).shape == (n_components, n_features)