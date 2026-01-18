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
@pytest.mark.parametrize('obj', [np.ones(3), np.ones(1, dtype='i,i')[()]])
def test_error_if_stored_buffer_info_is_corrupted(self, obj):
    """
        If a user extends a NumPy array before 1.20 and then runs it
        on NumPy 1.20+. A C-subclassed array might in theory modify
        the new buffer-info field. This checks that an error is raised
        if this happens (for buffer export), an error is written on delete.
        This is a sanity check to help users transition to safe code, it
        may be deleted at any point.
        """
    _multiarray_tests.corrupt_or_fix_bufferinfo(obj)
    name = type(obj)
    with pytest.raises(RuntimeError, match=f'.*{name} appears to be C subclassed'):
        memoryview(obj)
    _multiarray_tests.corrupt_or_fix_bufferinfo(obj)