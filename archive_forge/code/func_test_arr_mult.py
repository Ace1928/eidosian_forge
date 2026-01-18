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
@pytest.mark.parametrize('func', (np.dot, np.matmul))
def test_arr_mult(self, func):
    a = np.array([[1, 0], [0, 1]])
    b = np.array([[0, 1], [1, 0]])
    c = np.array([[9, 1], [1, -9]])
    d = np.arange(24).reshape(4, 6)
    ddt = np.array([[55, 145, 235, 325], [145, 451, 757, 1063], [235, 757, 1279, 1801], [325, 1063, 1801, 2539]])
    dtd = np.array([[504, 540, 576, 612, 648, 684], [540, 580, 620, 660, 700, 740], [576, 620, 664, 708, 752, 796], [612, 660, 708, 756, 804, 852], [648, 700, 752, 804, 856, 908], [684, 740, 796, 852, 908, 964]])
    for et in [np.float32, np.float64, np.complex64, np.complex128]:
        eaf = a.astype(et)
        assert_equal(func(eaf, eaf), eaf)
        assert_equal(func(eaf.T, eaf), eaf)
        assert_equal(func(eaf, eaf.T), eaf)
        assert_equal(func(eaf.T, eaf.T), eaf)
        assert_equal(func(eaf.T.copy(), eaf), eaf)
        assert_equal(func(eaf, eaf.T.copy()), eaf)
        assert_equal(func(eaf.T.copy(), eaf.T.copy()), eaf)
    for et in [np.float32, np.float64, np.complex64, np.complex128]:
        eaf = a.astype(et)
        ebf = b.astype(et)
        assert_equal(func(ebf, ebf), eaf)
        assert_equal(func(ebf.T, ebf), eaf)
        assert_equal(func(ebf, ebf.T), eaf)
        assert_equal(func(ebf.T, ebf.T), eaf)
    for et in [np.float32, np.float64, np.complex64, np.complex128]:
        edf = d.astype(et)
        assert_equal(func(edf[::-1, :], edf.T), func(edf[::-1, :].copy(), edf.T.copy()))
        assert_equal(func(edf[:, ::-1], edf.T), func(edf[:, ::-1].copy(), edf.T.copy()))
        assert_equal(func(edf, edf[::-1, :].T), func(edf, edf[::-1, :].T.copy()))
        assert_equal(func(edf, edf[:, ::-1].T), func(edf, edf[:, ::-1].T.copy()))
        assert_equal(func(edf[:edf.shape[0] // 2, :], edf[::2, :].T), func(edf[:edf.shape[0] // 2, :].copy(), edf[::2, :].T.copy()))
        assert_equal(func(edf[::2, :], edf[:edf.shape[0] // 2, :].T), func(edf[::2, :].copy(), edf[:edf.shape[0] // 2, :].T.copy()))
    for et in [np.float32, np.float64, np.complex64, np.complex128]:
        edf = d.astype(et)
        eddtf = ddt.astype(et)
        edtdf = dtd.astype(et)
        assert_equal(func(edf, edf.T), eddtf)
        assert_equal(func(edf.T, edf), edtdf)