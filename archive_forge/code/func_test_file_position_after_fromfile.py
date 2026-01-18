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
def test_file_position_after_fromfile(self, tmp_filename):
    sizes = [io.DEFAULT_BUFFER_SIZE // 8, io.DEFAULT_BUFFER_SIZE, io.DEFAULT_BUFFER_SIZE * 8]
    for size in sizes:
        with open(tmp_filename, 'wb') as f:
            f.seek(size - 1)
            f.write(b'\x00')
        for mode in ['rb', 'r+b']:
            err_msg = '%d %s' % (size, mode)
            with open(tmp_filename, mode) as f:
                f.read(2)
                np.fromfile(f, dtype=np.float64, count=1)
                pos = f.tell()
            assert_equal(pos, 10, err_msg=err_msg)