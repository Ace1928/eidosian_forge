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
def test_inner_product_with_various_contiguities(self):
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        A = np.array([[1, 2], [3, 4]], dtype=dt)
        B = np.array([[1, 3], [2, 4]], dtype=dt)
        C = np.array([1, 1], dtype=dt)
        desired = np.array([4, 6], dtype=dt)
        assert_equal(np.inner(A.T, C), desired)
        assert_equal(np.inner(C, A.T), desired)
        assert_equal(np.inner(B, C), desired)
        assert_equal(np.inner(C, B), desired)
        desired = np.array([[7, 10], [15, 22]], dtype=dt)
        assert_equal(np.inner(A, B), desired)
        desired = np.array([[5, 11], [11, 25]], dtype=dt)
        assert_equal(np.inner(A, A), desired)
        assert_equal(np.inner(A, A.copy()), desired)
        a = np.arange(5).astype(dt)
        b = a[::-1]
        desired = np.array(10, dtype=dt).item()
        assert_equal(np.inner(b, a), desired)