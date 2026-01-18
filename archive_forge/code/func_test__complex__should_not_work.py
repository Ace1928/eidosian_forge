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
def test__complex__should_not_work(self):
    dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f', 'd', 'g', 'F', 'D', 'G', '?', 'O']
    for dt in dtypes:
        a = np.array([1, 2, 3], dtype=dt)
        assert_raises(TypeError, complex, a)
    dt = np.dtype([('a', 'f8'), ('b', 'i1')])
    b = np.array((1.0, 3), dtype=dt)
    assert_raises(TypeError, complex, b)
    c = np.array([(1.0, 3), (0.002, 7)], dtype=dt)
    assert_raises(TypeError, complex, c)
    d = np.array('1+1j')
    assert_raises(TypeError, complex, d)
    e = np.array(['1+1j'], 'U')
    with assert_warns(DeprecationWarning):
        assert_raises(TypeError, complex, e)