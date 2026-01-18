import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_fourier_basis(self):
    np.testing.assert_allclose(self._G.e[0], 0, atol=1e-12)
    N = self._G.N
    np.testing.assert_allclose(self._G.U[:, 0], np.sqrt(N) / N)
    G = graphs.Logo(lap_type='normalized')
    G.compute_fourier_basis()
    assert G.e[-1] < 2