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
def test_3d_tensor(self):
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        a = np.arange(24).reshape(2, 3, 4).astype(dt)
        b = np.arange(24, 48).reshape(2, 3, 4).astype(dt)
        desired = np.array([[[[158, 182, 206], [230, 254, 278]], [[566, 654, 742], [830, 918, 1006]], [[974, 1126, 1278], [1430, 1582, 1734]]], [[[1382, 1598, 1814], [2030, 2246, 2462]], [[1790, 2070, 2350], [2630, 2910, 3190]], [[2198, 2542, 2886], [3230, 3574, 3918]]]]).astype(dt)
        assert_equal(np.inner(a, b), desired)
        assert_equal(np.inner(b, a).transpose(2, 3, 0, 1), desired)