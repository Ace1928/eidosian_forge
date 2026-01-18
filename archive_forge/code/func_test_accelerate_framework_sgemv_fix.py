from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
def test_accelerate_framework_sgemv_fix(self):

    def aligned_array(shape, align, dtype, order='C'):
        d = dtype(0)
        N = np.prod(shape)
        tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
        address = tmp.__array_interface__['data'][0]
        for offset in range(align):
            if (address + offset) % align == 0:
                break
        tmp = tmp[offset:offset + N * d.nbytes].view(dtype=dtype)
        return tmp.reshape(shape, order=order)

    def as_aligned(arr, align, dtype, order='C'):
        aligned = aligned_array(arr.shape, align, dtype, order)
        aligned[:] = arr[:]
        return aligned

    def assert_dot_close(A, X, desired):
        assert_allclose(np.dot(A, X), desired, rtol=1e-05, atol=1e-07)
    m = aligned_array(100, 15, np.float32)
    s = aligned_array((100, 100), 15, np.float32)
    np.dot(s, m)
    testdata = itertools.product((15, 32), (10000,), (200, 89), ('C', 'F'))
    for align, m, n, a_order in testdata:
        A_d = np.random.rand(m, n)
        X_d = np.random.rand(n)
        desired = np.dot(A_d, X_d)
        A_f = as_aligned(A_d, align, np.float32, order=a_order)
        X_f = as_aligned(X_d, align, np.float32)
        assert_dot_close(A_f, X_f, desired)
        A_d_2 = A_d[::2]
        desired = np.dot(A_d_2, X_d)
        A_f_2 = A_f[::2]
        assert_dot_close(A_f_2, X_f, desired)
        A_d_22 = A_d_2[:, ::2]
        X_d_2 = X_d[::2]
        desired = np.dot(A_d_22, X_d_2)
        A_f_22 = A_f_2[:, ::2]
        X_f_2 = X_f[::2]
        assert_dot_close(A_f_22, X_f_2, desired)
        if a_order == 'F':
            assert_equal(A_f_22.strides, (8, 8 * m))
        else:
            assert_equal(A_f_22.strides, (8 * n, 8))
        assert_equal(X_f_2.strides, (8,))
        X_f_2c = as_aligned(X_f_2, align, np.float32)
        assert_dot_close(A_f_22, X_f_2c, desired)
        A_d_12 = A_d[:, ::2]
        desired = np.dot(A_d_12, X_d_2)
        A_f_12 = A_f[:, ::2]
        assert_dot_close(A_f_12, X_f_2c, desired)
        assert_dot_close(A_f_12, X_f_2, desired)