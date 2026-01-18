import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_economic_check_finite(self):
    a0, q0, r0, u0, v0 = self.generate('tall', mode='economic', p=3)
    q = q0.copy('F')
    q[1, 1] = np.nan
    assert_raises(ValueError, qr_update, q, r0, u0[:, 0], v0[:, 0])
    assert_raises(ValueError, qr_update, q, r0, u0, v0)
    r = r0.copy('F')
    r[1, 1] = np.nan
    assert_raises(ValueError, qr_update, q0, r, u0[:, 0], v0[:, 0])
    assert_raises(ValueError, qr_update, q0, r, u0, v0)
    u = u0.copy('F')
    u[0, 0] = np.nan
    assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v0[:, 0])
    assert_raises(ValueError, qr_update, q0, r0, u, v0)
    v = v0.copy('F')
    v[0, 0] = np.nan
    assert_raises(ValueError, qr_update, q0, r0, u[:, 0], v[:, 0])
    assert_raises(ValueError, qr_update, q0, r0, u, v)