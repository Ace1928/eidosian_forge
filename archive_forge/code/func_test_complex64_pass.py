import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_complex64_pass(self):
    nulp = 5
    x = np.linspace(-20, 20, 50, dtype=np.float32)
    x = 10 ** x
    x = np.r_[-x, x]
    xi = x + x * 1j
    eps = np.finfo(x.dtype).eps
    y = x + x * eps * nulp / 2.0
    assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
    assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
    y = x + x * eps * nulp / 4.0
    assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)
    epsneg = np.finfo(x.dtype).epsneg
    y = x - x * epsneg * nulp / 2.0
    assert_array_almost_equal_nulp(xi, x + y * 1j, nulp)
    assert_array_almost_equal_nulp(xi, y + x * 1j, nulp)
    y = x - x * epsneg * nulp / 4.0
    assert_array_almost_equal_nulp(xi, y + y * 1j, nulp)