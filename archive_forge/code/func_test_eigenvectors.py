import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import diags, csgraph
from scipy.linalg import eigh
from scipy.sparse.linalg import LaplacianNd
from scipy.sparse.linalg._special_sparse_arrays import Sakurai
from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
@pytest.mark.parametrize('grid_shape', [(6,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize('bc', ['neumann', 'dirichlet', 'periodic'])
def test_eigenvectors(self, grid_shape, bc):
    lap = LaplacianNd(grid_shape, boundary_conditions=bc, dtype=np.float64)
    n = np.prod(grid_shape)
    eigenvalues = lap.eigenvalues()
    eigenvectors = lap.eigenvectors()
    dtype = eigenvectors.dtype
    atol = n * n * max(np.finfo(dtype).eps, np.finfo(np.double).eps)
    for i in np.arange(n):
        r = lap.toarray() @ eigenvectors[:, i] - eigenvectors[:, i] * eigenvalues[i]
        assert_allclose(r, np.zeros_like(r), atol=atol)
    for m in np.arange(1, n + 1):
        e = lap.eigenvalues(m)
        ev = lap.eigenvectors(m)
        r = lap.toarray() @ ev - ev @ np.diag(e)
        assert_allclose(r, np.zeros_like(r), atol=atol)