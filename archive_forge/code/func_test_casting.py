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
def test_casting(self):
    a = np.array([(1,)], dtype=[('a', '<i4')])
    assert_(np.can_cast(a.dtype, [('a', '>i4')], casting='unsafe'))
    b = a.astype([('a', '>i4')])
    assert_equal(b, a.byteswap().newbyteorder())
    assert_equal(a['a'][0], b['a'][0])
    a = np.array([(5, 42), (10, 1)], dtype=[('a', '>i4'), ('b', '<f8')])
    b = np.array([(5, 42), (10, 1)], dtype=[('a', '<i4'), ('b', '>f8')])
    assert_(np.can_cast(a.dtype, b.dtype, casting='equiv'))
    assert_equal(a == b, [True, True])
    assert_(np.can_cast(a.dtype, b.dtype, casting='equiv'))
    c = a.astype(b.dtype, casting='equiv')
    assert_equal(a == c, [True, True])
    t = [('a', '<i8'), ('b', '>f8')]
    assert_(np.can_cast(a.dtype, t, casting='safe'))
    c = a.astype(t, casting='safe')
    assert_equal(c == np.array([(5, 42), (10, 1)], dtype=t), [True, True])
    t = [('a', '<i4'), ('b', '>f4')]
    assert_(np.can_cast(a.dtype, t, casting='same_kind'))
    c = a.astype(t, casting='same_kind')
    assert_equal(c == np.array([(5, 42), (10, 1)], dtype=t), [True, True])
    t = [('a', '>i8'), ('b', '<f4')]
    assert_(not np.can_cast(a.dtype, t, casting='safe'))
    assert_raises(TypeError, a.astype, t, casting='safe')
    t = [('a', '>i2'), ('b', '<f8')]
    assert_(not np.can_cast(a.dtype, t, casting='equiv'))
    assert_raises(TypeError, a.astype, t, casting='equiv')
    t = [('a', '>i8'), ('b', '<i2')]
    assert_(not np.can_cast(a.dtype, t, casting='same_kind'))
    assert_raises(TypeError, a.astype, t, casting='same_kind')
    assert_(not np.can_cast(a.dtype, b.dtype, casting='no'))
    assert_raises(TypeError, a.astype, b.dtype, casting='no')
    for casting in ['no', 'safe', 'equiv', 'same_kind']:
        t = [('a', '>i4')]
        assert_(not np.can_cast(a.dtype, t, casting=casting))
        t = [('a', '>i4'), ('b', '<f8'), ('c', 'i4')]
        assert_(not np.can_cast(a.dtype, t, casting=casting))