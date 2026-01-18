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
def test_mean_where(self):
    a = np.arange(16).reshape((4, 4))
    wh_full = np.array([[False, True, False, True], [True, False, True, False], [True, True, False, False], [False, False, True, True]])
    wh_partial = np.array([[False], [True], [True], [False]])
    _cases = [(1, True, [1.5, 5.5, 9.5, 13.5]), (0, wh_full, [6.0, 5.0, 10.0, 9.0]), (1, wh_full, [2.0, 5.0, 8.5, 14.5]), (0, wh_partial, [6.0, 7.0, 8.0, 9.0])]
    for _ax, _wh, _res in _cases:
        assert_allclose(a.mean(axis=_ax, where=_wh), np.array(_res))
        assert_allclose(np.mean(a, axis=_ax, where=_wh), np.array(_res))
    a3d = np.arange(16).reshape((2, 2, 4))
    _wh_partial = np.array([False, True, True, False])
    _res = [[1.5, 5.5], [9.5, 13.5]]
    assert_allclose(a3d.mean(axis=2, where=_wh_partial), np.array(_res))
    assert_allclose(np.mean(a3d, axis=2, where=_wh_partial), np.array(_res))
    with pytest.warns(RuntimeWarning) as w:
        assert_allclose(a.mean(axis=1, where=wh_partial), np.array([np.nan, 5.5, 9.5, np.nan]))
    with pytest.warns(RuntimeWarning) as w:
        assert_equal(a.mean(where=False), np.nan)
    with pytest.warns(RuntimeWarning) as w:
        assert_equal(np.mean(a, where=False), np.nan)