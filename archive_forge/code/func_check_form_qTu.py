import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def check_form_qTu(q_order, q_shape, u_order, u_shape, u_ndim, dtype):
    np.random.seed(47)
    if u_shape == 1 and u_ndim == 1:
        u_shape = (q_shape[0],)
    else:
        u_shape = (q_shape[0], u_shape)
    dtype = np.dtype(dtype)
    if dtype.char in 'fd':
        q = np.random.random(q_shape)
        u = np.random.random(u_shape)
    elif dtype.char in 'FD':
        q = np.random.random(q_shape) + 1j * np.random.random(q_shape)
        u = np.random.random(u_shape) + 1j * np.random.random(u_shape)
    else:
        ValueError("form_qTu doesn't support this dtype")
    q = np.require(q, dtype, q_order)
    if u_order != 'A':
        u = np.require(u, dtype, u_order)
    else:
        u, = make_strided((u.astype(dtype),))
    rtol = 10.0 ** (-(np.finfo(dtype).precision - 2))
    atol = 2 * np.finfo(dtype).eps
    expected = np.dot(q.T.conj(), u)
    res = _decomp_update._form_qTu(q, u)
    assert_allclose(res, expected, rtol=rtol, atol=atol)