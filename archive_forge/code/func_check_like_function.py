import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def check_like_function(self, like_function, value, fill_value=False):
    if fill_value:
        fill_kwarg = {'fill_value': value}
    else:
        fill_kwarg = {}
    for d, dtype in self.data:
        dz = like_function(d, dtype=dtype, **fill_kwarg)
        assert_equal(dz.shape, d.shape)
        assert_equal(np.array(dz.strides) * d.dtype.itemsize, np.array(d.strides) * dz.dtype.itemsize)
        assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
        assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
        if dtype is None:
            assert_equal(dz.dtype, d.dtype)
        else:
            assert_equal(dz.dtype, np.dtype(dtype))
        self.compare_array_value(dz, value, fill_value)
        dz = like_function(d, order='C', dtype=dtype, **fill_kwarg)
        assert_equal(dz.shape, d.shape)
        assert_(dz.flags.c_contiguous)
        if dtype is None:
            assert_equal(dz.dtype, d.dtype)
        else:
            assert_equal(dz.dtype, np.dtype(dtype))
        self.compare_array_value(dz, value, fill_value)
        dz = like_function(d, order='F', dtype=dtype, **fill_kwarg)
        assert_equal(dz.shape, d.shape)
        assert_(dz.flags.f_contiguous)
        if dtype is None:
            assert_equal(dz.dtype, d.dtype)
        else:
            assert_equal(dz.dtype, np.dtype(dtype))
        self.compare_array_value(dz, value, fill_value)
        dz = like_function(d, order='A', dtype=dtype, **fill_kwarg)
        assert_equal(dz.shape, d.shape)
        if d.flags.f_contiguous:
            assert_(dz.flags.f_contiguous)
        else:
            assert_(dz.flags.c_contiguous)
        if dtype is None:
            assert_equal(dz.dtype, d.dtype)
        else:
            assert_equal(dz.dtype, np.dtype(dtype))
        self.compare_array_value(dz, value, fill_value)
        for s in self.shapes:
            for o in 'CFA':
                sz = like_function(d, dtype=dtype, shape=s, order=o, **fill_kwarg)
                assert_equal(sz.shape, s)
                if dtype is None:
                    assert_equal(sz.dtype, d.dtype)
                else:
                    assert_equal(sz.dtype, np.dtype(dtype))
                if o == 'C' or (o == 'A' and d.flags.c_contiguous):
                    assert_(sz.flags.c_contiguous)
                elif o == 'F' or (o == 'A' and d.flags.f_contiguous):
                    assert_(sz.flags.f_contiguous)
                self.compare_array_value(sz, value, fill_value)
            if d.ndim != len(s):
                assert_equal(np.argsort(like_function(d, dtype=dtype, shape=s, order='K', **fill_kwarg).strides), np.argsort(np.empty(s, dtype=dtype, order='C').strides))
            else:
                assert_equal(np.argsort(like_function(d, dtype=dtype, shape=s, order='K', **fill_kwarg).strides), np.argsort(d.strides))

    class MyNDArray(np.ndarray):
        pass
    a = np.array([[1, 2], [3, 4]]).view(MyNDArray)
    b = like_function(a, **fill_kwarg)
    assert_(type(b) is MyNDArray)
    b = like_function(a, subok=False, **fill_kwarg)
    assert_(type(b) is not MyNDArray)