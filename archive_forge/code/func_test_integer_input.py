import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_integer_input(self):
    q = np.arange(16).reshape(4, 4)
    r = q.copy()
    u = q[:, 0].copy()
    v = r[0, :].copy()
    assert_raises(ValueError, qr_update, q, r, u, v)