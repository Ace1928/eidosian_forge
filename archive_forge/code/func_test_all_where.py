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
def test_all_where(self):
    a = np.array([[True, False, True], [False, False, False], [True, True, True]])
    wh_full = np.array([[True, False, True], [False, False, False], [True, False, True]])
    wh_lower = np.array([[False], [False], [True]])
    for _ax in [0, None]:
        assert_equal(a.all(axis=_ax, where=wh_lower), np.all(a[wh_lower[:, 0], :], axis=_ax))
        assert_equal(np.all(a, axis=_ax, where=wh_lower), a[wh_lower[:, 0], :].all(axis=_ax))
    assert_equal(a.all(where=wh_full), True)
    assert_equal(np.all(a, where=wh_full), True)
    assert_equal(a.all(where=False), True)
    assert_equal(np.all(a, where=False), True)