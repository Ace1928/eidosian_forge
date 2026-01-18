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
def test_field_names(self):
    a = np.zeros((1,), dtype=[('f1', 'i4'), ('f2', 'i4'), ('f3', [('sf1', 'i4')])])
    assert_raises(IndexError, a.__setitem__, b'f1', 1)
    assert_raises(IndexError, a.__getitem__, b'f1')
    assert_raises(IndexError, a['f1'].__setitem__, b'sf1', 1)
    assert_raises(IndexError, a['f1'].__getitem__, b'sf1')
    b = a.copy()
    fn1 = str('f1')
    b[fn1] = 1
    assert_equal(b[fn1], 1)
    fnn = str('not at all')
    assert_raises(ValueError, b.__setitem__, fnn, 1)
    assert_raises(ValueError, b.__getitem__, fnn)
    b[0][fn1] = 2
    assert_equal(b[fn1], 2)
    assert_raises(ValueError, b[0].__setitem__, fnn, 1)
    assert_raises(ValueError, b[0].__getitem__, fnn)
    fn3 = str('f3')
    sfn1 = str('sf1')
    b[fn3][sfn1] = 1
    assert_equal(b[fn3][sfn1], 1)
    assert_raises(ValueError, b[fn3].__setitem__, fnn, 1)
    assert_raises(ValueError, b[fn3].__getitem__, fnn)
    fn2 = str('f2')
    b[fn2] = 3
    assert_equal(b[['f1', 'f2']][0].tolist(), (2, 3))
    assert_equal(b[['f2', 'f1']][0].tolist(), (3, 2))
    assert_equal(b[['f1', 'f3']][0].tolist(), (2, (1,)))
    assert_raises(ValueError, a.__setitem__, 'Ϡ', 1)
    assert_raises(ValueError, a.__getitem__, 'Ϡ')