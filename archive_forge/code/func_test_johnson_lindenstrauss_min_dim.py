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
def test_johnson_lindenstrauss_min_dim():
    """Test Johnson-Lindenstrauss for small eps.

    Regression test for #17111: before #19374, 32-bit systems would fail.
    """
    assert johnson_lindenstrauss_min_dim(100, eps=1e-05) == 368416070986