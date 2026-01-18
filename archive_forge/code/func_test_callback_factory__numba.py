from __future__ import (absolute_import, division, print_function)
from .._sympy_Lambdify import _callback_factory
from sympy import symbols, atan
import numpy as np
import pytest
@pytest.mark.skipif(numba is None, reason='numba not available')
def test_callback_factory__numba():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = _callback_factory(args, [expr], 'numpy', np.float64, 'C', use_numba=True)
    n = 500
    inp = np.empty((n, 2))
    inp[:, 0] = np.linspace(0, 1, n)
    inp[:, 1] = np.linspace(-10, 10, n)
    assert np.allclose(cb(inp), inp[:, 0] + np.arctan(inp[:, 1]))