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
def test_file_position_after_tofile(self, tmp_filename):
    sizes = [io.DEFAULT_BUFFER_SIZE // 8, io.DEFAULT_BUFFER_SIZE, io.DEFAULT_BUFFER_SIZE * 8]
    for size in sizes:
        err_msg = '%d' % (size,)
        with open(tmp_filename, 'wb') as f:
            f.seek(size - 1)
            f.write(b'\x00')
            f.seek(10)
            f.write(b'12')
            np.array([0], dtype=np.float64).tofile(f)
            pos = f.tell()
        assert_equal(pos, 10 + 2 + 8, err_msg=err_msg)
        with open(tmp_filename, 'r+b') as f:
            f.read(2)
            f.seek(0, 1)
            np.array([0], dtype=np.float64).tofile(f)
            pos = f.tell()
        assert_equal(pos, 10, err_msg=err_msg)