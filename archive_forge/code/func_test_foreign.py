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
def test_foreign(self):
    c = np.array([False, True, False, False, False, False, True, False, False, False, True, False])
    r = np.array([5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0], dtype=np.float64)
    a = np.ones(1, dtype='>i4')
    b = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0], dtype=np.float64)
    assert_equal(np.where(c, a, b), r)
    b = b.astype('>f8')
    assert_equal(np.where(c, a, b), r)
    a = a.astype('<i4')
    assert_equal(np.where(c, a, b), r)
    c = c.astype('>i4')
    assert_equal(np.where(c, a, b), r)