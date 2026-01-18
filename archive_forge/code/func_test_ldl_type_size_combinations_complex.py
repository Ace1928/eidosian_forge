from numpy.testing import assert_array_almost_equal, assert_allclose, assert_
from numpy import (array, eye, zeros, empty_like, empty, tril_indices_from,
from numpy.random import rand, randint, seed
from scipy.linalg import ldl
from scipy._lib._util import ComplexWarning
import pytest
from pytest import raises as assert_raises, warns
@pytest.mark.parametrize('dtype', [complex64, complex128])
@pytest.mark.parametrize('n', [30, 150])
def test_ldl_type_size_combinations_complex(n, dtype):
    seed(1234)
    msg1 = f'Her failed for size: {n}, dtype: {dtype}'
    msg2 = f'Sym failed for size: {n}, dtype: {dtype}'
    x = (rand(n, n) + 1j * rand(n, n)).astype(dtype)
    x = x + x.conj().T
    x += eye(n, dtype=dtype) * dtype(randint(5, 1000000.0))
    l, d1, p = ldl(x)
    u, d2, p = ldl(x, lower=0)
    rtol = 0.0002 if dtype is complex64 else 1e-10
    assert_allclose(l.dot(d1).dot(l.conj().T), x, rtol=rtol, err_msg=msg1)
    assert_allclose(u.dot(d2).dot(u.conj().T), x, rtol=rtol, err_msg=msg1)
    x = (rand(n, n) + 1j * rand(n, n)).astype(dtype)
    x = x + x.T
    x += eye(n, dtype=dtype) * dtype(randint(5, 1000000.0))
    l, d1, p = ldl(x, hermitian=0)
    u, d2, p = ldl(x, lower=0, hermitian=0)
    assert_allclose(l.dot(d1).dot(l.T), x, rtol=rtol, err_msg=msg2)
    assert_allclose(u.dot(d2).dot(u.T), x, rtol=rtol, err_msg=msg2)