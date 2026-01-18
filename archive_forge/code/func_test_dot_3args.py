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
def test_dot_3args(self):
    from numpy.core.multiarray import dot
    np.random.seed(22)
    f = np.random.random_sample((1024, 16))
    v = np.random.random_sample((16, 32))
    r = np.empty((1024, 32))
    for i in range(12):
        dot(f, v, r)
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(r), 2)
    r2 = dot(f, v, out=None)
    assert_array_equal(r2, r)
    assert_(r is dot(f, v, out=r))
    v = v[:, 0].copy()
    r = r[:, 0].copy()
    r2 = dot(f, v)
    assert_(r is dot(f, v, r))
    assert_array_equal(r2, r)