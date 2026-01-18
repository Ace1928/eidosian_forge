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
def test_assign_mask2(self):
    xorig = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    m = np.array([0, 1], bool)
    m2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], bool)
    m3 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
    x = xorig.copy()
    x[m] = 10
    assert_array_equal(x, np.array([[1, 2, 3, 4], [10, 10, 10, 10]]))
    x = xorig.copy()
    x[m2] = 10
    assert_array_equal(x, np.array([[1, 10, 3, 4], [10, 6, 7, 8]]))
    x = xorig.copy()
    x[m3] = 10
    assert_array_equal(x, np.array([[1, 10, 3, 4], [5, 6, 7, 8]]))