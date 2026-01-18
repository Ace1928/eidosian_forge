import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_mixed_types():
    seed(1234)
    x = np.array(ortho_group.rvs(4), dtype=np.float64)
    u, cs, vh = cossin([x[:2, :2], np.array(x[:2, 2:], dtype=np.complex128), x[2:, :2], x[2:, 2:]])
    assert u.dtype == np.complex128
    assert cs.dtype == np.float64
    assert vh.dtype == np.complex128
    assert_allclose(x, u @ cs @ vh, rtol=0.0, atol=10000.0 * np.finfo(np.complex128).eps)