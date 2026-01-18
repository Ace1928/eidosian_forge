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
def test_largish_file(self, tmp_filename):
    d = np.zeros(4 * 1024 ** 2)
    d.tofile(tmp_filename)
    assert_equal(os.path.getsize(tmp_filename), d.nbytes)
    assert_array_equal(d, np.fromfile(tmp_filename))
    with open(tmp_filename, 'r+b') as f:
        f.seek(d.nbytes)
        d.tofile(f)
        assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)
    open(tmp_filename, 'w').close()
    with open(tmp_filename, 'ab') as f:
        d.tofile(f)
    assert_array_equal(d, np.fromfile(tmp_filename))
    with open(tmp_filename, 'ab') as f:
        d.tofile(f)
    assert_equal(os.path.getsize(tmp_filename), d.nbytes * 2)