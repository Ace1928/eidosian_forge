import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_axis_argument(self):
    inner1d = umt.inner1d
    a = np.arange(27.0).reshape((3, 3, 3))
    b = np.arange(10.0, 19.0).reshape((3, 1, 3))
    c = inner1d(a, b)
    assert_array_equal(c, (a * b).sum(-1))
    c = inner1d(a, b, axis=-1)
    assert_array_equal(c, (a * b).sum(-1))
    out = np.zeros_like(c)
    d = inner1d(a, b, axis=-1, out=out)
    assert_(d is out)
    assert_array_equal(d, c)
    c = inner1d(a, b, axis=0)
    assert_array_equal(c, (a * b).sum(0))
    a = np.arange(6).reshape((2, 3))
    b = np.arange(10, 16).reshape((2, 3))
    w = np.arange(20, 26).reshape((2, 3))
    assert_array_equal(umt.innerwt(a, b, w, axis=0), np.sum(a * b * w, axis=0))
    assert_array_equal(umt.cumsum(a, axis=0), np.cumsum(a, axis=0))
    assert_array_equal(umt.cumsum(a, axis=-1), np.cumsum(a, axis=-1))
    out = np.empty_like(a)
    b = umt.cumsum(a, out=out, axis=0)
    assert_(out is b)
    assert_array_equal(b, np.cumsum(a, axis=0))
    b = umt.cumsum(a, out=out, axis=1)
    assert_(out is b)
    assert_array_equal(b, np.cumsum(a, axis=-1))
    assert_raises(TypeError, inner1d, a, b, axis=0, axes=[0, 0])
    assert_raises(TypeError, inner1d, a, b, axis=[0])
    mm = umt.matrix_multiply
    assert_raises(TypeError, mm, a, b, axis=1)
    out = np.empty((1, 2, 3), dtype=a.dtype)
    assert_raises(ValueError, umt.cumsum, a, out=out, axis=0)
    assert_raises(TypeError, np.add, 1.0, 1.0, axis=0)