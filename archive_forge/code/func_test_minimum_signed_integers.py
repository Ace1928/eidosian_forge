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
def test_minimum_signed_integers(self):
    a = np.array([1, -2 ** 7, -2 ** 7 + 1, 2 ** 7 - 1], dtype=np.int8)
    assert_equal(np.argmin(a), 1)
    a = a.repeat(129)
    assert_equal(np.argmin(a), 129)
    a = np.array([1, -2 ** 15, -2 ** 15 + 1, 2 ** 15 - 1], dtype=np.int16)
    assert_equal(np.argmin(a), 1)
    a = a.repeat(129)
    assert_equal(np.argmin(a), 129)
    a = np.array([1, -2 ** 31, -2 ** 31 + 1, 2 ** 31 - 1], dtype=np.int32)
    assert_equal(np.argmin(a), 1)
    a = a.repeat(129)
    assert_equal(np.argmin(a), 129)
    a = np.array([1, -2 ** 63, -2 ** 63 + 1, 2 ** 63 - 1], dtype=np.int64)
    assert_equal(np.argmin(a), 1)
    a = a.repeat(129)
    assert_equal(np.argmin(a), 129)