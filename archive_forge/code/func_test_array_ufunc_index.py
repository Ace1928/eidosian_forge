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
def test_array_ufunc_index(self):

    class CheckIndex:

        def __array_ufunc__(self, ufunc, method, *inputs, **kw):
            for i, a in enumerate(inputs):
                if a is self:
                    return i
            for j, a in enumerate(kw['out']):
                if a is self:
                    return (j,)
    a = CheckIndex()
    dummy = np.arange(2.0)
    assert_equal(np.sin(a), 0)
    assert_equal(np.sin(dummy, a), (0,))
    assert_equal(np.sin(dummy, out=a), (0,))
    assert_equal(np.sin(dummy, out=(a,)), (0,))
    assert_equal(np.sin(a, a), 0)
    assert_equal(np.sin(a, out=a), 0)
    assert_equal(np.sin(a, out=(a,)), 0)
    assert_equal(np.modf(dummy, a), (0,))
    assert_equal(np.modf(dummy, None, a), (1,))
    assert_equal(np.modf(dummy, dummy, a), (1,))
    assert_equal(np.modf(dummy, out=(a, None)), (0,))
    assert_equal(np.modf(dummy, out=(a, dummy)), (0,))
    assert_equal(np.modf(dummy, out=(None, a)), (1,))
    assert_equal(np.modf(dummy, out=(dummy, a)), (1,))
    assert_equal(np.modf(a, out=(dummy, a)), 0)
    with assert_raises(TypeError):
        np.modf(dummy, out=a)
    assert_raises(ValueError, np.modf, dummy, out=(a,))
    assert_equal(np.add(a, dummy), 0)
    assert_equal(np.add(dummy, a), 1)
    assert_equal(np.add(dummy, dummy, a), (0,))
    assert_equal(np.add(dummy, a, a), 1)
    assert_equal(np.add(dummy, dummy, out=a), (0,))
    assert_equal(np.add(dummy, dummy, out=(a,)), (0,))
    assert_equal(np.add(a, dummy, out=a), 0)