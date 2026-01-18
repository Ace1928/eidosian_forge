import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
def test_1d_with_graph_laplacian(self):
    n = 6
    G = diags(np.ones(n - 1), 1, format='dia')
    Lf = csgraph.laplacian(G, symmetrized=True, form='function')
    La = csgraph.laplacian(G, symmetrized=True, form='array')
    grid_shape = (n,)
    bc = 'neumann'
    lap = LaplacianNd(grid_shape, boundary_conditions=bc)
    assert_array_equal(lap(np.eye(n)), -Lf(np.eye(n)))
    assert_array_equal(lap.toarray(), -La.toarray())
    assert_array_equal(lap.tosparse().toarray(), -La.toarray())