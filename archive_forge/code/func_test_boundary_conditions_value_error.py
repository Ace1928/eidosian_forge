import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
def test_boundary_conditions_value_error(self):
    with pytest.raises(ValueError, match="Unknown value 'robin'"):
        LaplacianNd(grid_shape=(6,), boundary_conditions='robin')