import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
def test_half_funcs(self):
    """Test the various ArrFuncs"""
    assert_equal(np.arange(10, dtype=float16), np.arange(10, dtype=float32))
    a = np.zeros((5,), dtype=float16)
    a.fill(1)
    assert_equal(a, np.ones((5,), dtype=float16))
    a = np.array([0, 0, -1, -1 / 1e+20, 0, 2.0 ** (-24), 7.629e-06], dtype=float16)
    assert_equal(a.nonzero()[0], [2, 5, 6])
    a = a.byteswap()
    a = a.view(a.dtype.newbyteorder())
    assert_equal(a.nonzero()[0], [2, 5, 6])
    a = np.arange(0, 10, 0.5, dtype=float16)
    b = np.ones((20,), dtype=float16)
    assert_equal(np.dot(a, b), 95)
    a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
    assert_equal(a.argmax(), 4)
    a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
    assert_equal(a.argmax(), 5)
    a = np.arange(10, dtype=float16)
    for i in range(10):
        assert_equal(a.item(i), i)