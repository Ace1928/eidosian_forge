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
def test_dot_3args_errors(self):
    from numpy.core.multiarray import dot
    np.random.seed(22)
    f = np.random.random_sample((1024, 16))
    v = np.random.random_sample((16, 32))
    r = np.empty((1024, 31))
    assert_raises(ValueError, dot, f, v, r)
    r = np.empty((1024,))
    assert_raises(ValueError, dot, f, v, r)
    r = np.empty((32,))
    assert_raises(ValueError, dot, f, v, r)
    r = np.empty((32, 1024))
    assert_raises(ValueError, dot, f, v, r)
    assert_raises(ValueError, dot, f, v, r.T)
    r = np.empty((1024, 64))
    assert_raises(ValueError, dot, f, v, r[:, ::2])
    assert_raises(ValueError, dot, f, v, r[:, :32])
    r = np.empty((1024, 32), dtype=np.float32)
    assert_raises(ValueError, dot, f, v, r)
    r = np.empty((1024, 32), dtype=int)
    assert_raises(ValueError, dot, f, v, r)