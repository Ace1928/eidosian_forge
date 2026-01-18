import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def assert_upper_tri(a, rtol=None, atol=None):
    if rtol is None:
        rtol = 10.0 ** (-(np.finfo(a.dtype).precision - 2))
    if atol is None:
        atol = 2 * np.finfo(a.dtype).eps
    mask = np.tri(a.shape[0], a.shape[1], -1, np.bool_)
    assert_allclose(a[mask], 0.0, rtol=rtol, atol=atol)