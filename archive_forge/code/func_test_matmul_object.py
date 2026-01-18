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
def test_matmul_object(self):
    import fractions
    f = np.vectorize(fractions.Fraction)

    def random_ints():
        return np.random.randint(1, 1000, size=(10, 3, 3))
    M1 = f(random_ints(), random_ints())
    M2 = f(random_ints(), random_ints())
    M3 = self.matmul(M1, M2)
    [N1, N2, N3] = [a.astype(float) for a in [M1, M2, M3]]
    assert_allclose(N3, self.matmul(N1, N2))