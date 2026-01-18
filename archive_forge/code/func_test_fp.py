import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_fp(self):
    np.random.seed(1)

    def func(p, x):
        return p[0] + p[1] * x

    def err(p, x, y):
        return func(p, x) - y
    x = np.linspace(0, 1, 100, dtype=np.float64)
    y = np.random.random(100).astype(np.float64)
    p0 = np.array([-1.0, -1.0])
    jac_fp64 = approx_derivative(err, p0, method='2-point', args=(x, y))
    jac_fp = approx_derivative(err, p0.astype(np.float32), method='2-point', args=(x, y))
    assert err(p0, x, y).dtype == np.float64
    assert_allclose(jac_fp, jac_fp64, atol=0.001)

    def err_fp32(p):
        assert p.dtype == np.float32
        return err(p, x, y).astype(np.float32)
    jac_fp = approx_derivative(err_fp32, p0.astype(np.float32), method='2-point')
    assert_allclose(jac_fp, jac_fp64, atol=0.001)

    def f(x):
        return np.sin(x)

    def g(x):
        return np.cos(x)

    def hess(x):
        return -np.sin(x)

    def calc_atol(h, x0, f, hess, EPS):
        t0 = h / 2 * max(np.abs(hess(x0)), np.abs(hess(x0 + h)))
        t1 = EPS / h * max(np.abs(f(x0)), np.abs(f(x0 + h)))
        return t0 + t1
    for dtype in [np.float16, np.float32, np.float64]:
        EPS = np.finfo(dtype).eps
        x0 = np.array(1.0).astype(dtype)
        h = _compute_absolute_step(None, x0, f(x0), '2-point')
        atol = calc_atol(h, x0, f, hess, EPS)
        err = approx_derivative(f, x0, method='2-point', abs_step=h) - g(x0)
        assert abs(err) < atol