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
def test_array_interface_empty_shape():
    arr = np.array([1, 2, 3])
    interface1 = dict(arr.__array_interface__)
    interface1['shape'] = ()

    class DummyArray1:
        __array_interface__ = interface1
    interface2 = dict(interface1)
    interface2['data'] = arr[0].tobytes()

    class DummyArray2:
        __array_interface__ = interface2
    arr1 = np.asarray(DummyArray1())
    arr2 = np.asarray(DummyArray2())
    arr3 = arr[:1].reshape(())
    assert_equal(arr1, arr2)
    assert_equal(arr1, arr3)