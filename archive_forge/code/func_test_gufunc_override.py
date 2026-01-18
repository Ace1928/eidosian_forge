import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_gufunc_override(self):

    class A:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return (self, ufunc, method, inputs, kwargs)
    inner1d = ncu_tests.inner1d
    a = A()
    res = inner1d(a, a)
    assert_equal(res[0], a)
    assert_equal(res[1], inner1d)
    assert_equal(res[2], '__call__')
    assert_equal(res[3], (a, a))
    assert_equal(res[4], {})
    res = inner1d(1, 1, out=a)
    assert_equal(res[0], a)
    assert_equal(res[1], inner1d)
    assert_equal(res[2], '__call__')
    assert_equal(res[3], (1, 1))
    assert_equal(res[4], {'out': (a,)})
    assert_raises(TypeError, inner1d, a, out='two')
    assert_raises(TypeError, inner1d, a, a, 'one', out='two')
    assert_raises(TypeError, inner1d, a, a, 'one', 'two')
    assert_raises(ValueError, inner1d, a, a, out=('one', 'two'))
    assert_raises(ValueError, inner1d, a, a, out=())