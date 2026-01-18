import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
def test_matrix_norm_axis(self):
    for m, axis in ((self.b, None), (self.b, (0, 1)), (self.b.T, (1, 0))):
        assert_allclose(spnorm(m, axis=axis), 7.745966692414834)
        assert_allclose(spnorm(m, 'fro', axis=axis), 7.745966692414834)
        assert_allclose(spnorm(m, np.inf, axis=axis), 9)
        assert_allclose(spnorm(m, -np.inf, axis=axis), 2)
        assert_allclose(spnorm(m, 1, axis=axis), 7)
        assert_allclose(spnorm(m, -1, axis=axis), 6)