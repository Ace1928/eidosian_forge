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
@pytest.mark.parametrize('args', ((m1, m2), (m2.T, m1.T), (m2.T.copy(), m1.T), (m2.T, m1.T.copy()), (m1, m1.T), (m1.T, m1), (m1, m3.T), (m3, m1.T), (m3, m3.T), (m3.T, m3), (m3, m2), (m2.T, m3.T), (m2.T.copy(), m3.T), (m1, vr[:3]), (vc[:5], m1), (m1.T, vc[:5]), (vr[:3], m1.T), (m1, vr[::2]), (vc[::2], m1), (m1.T, vc[::2]), (vr[::2], m1.T), (m3, vr[:3]), (vc[:5], m3), (m3.T, vc[:5]), (vr[:3], m3.T), (m3, vr[::2]), (vc[::2], m3), (m3.T, vc[::2]), (vr[::2], m3.T), (m0, m0.T), (m0.T, m0), (m1, m0), (m0.T, m1.T)))
def test_dot_equivalent(self, args):
    r1 = np.matmul(*args)
    r2 = np.dot(*args)
    assert_equal(r1, r2)
    r3 = np.matmul(args[0].copy(), args[1].copy())
    assert_equal(r1, r3)