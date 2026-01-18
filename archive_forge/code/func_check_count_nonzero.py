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
def check_count_nonzero(self, power, length):
    powers = [2 ** i for i in range(length)]
    for i in range(2 ** power):
        l = [i & x != 0 for x in powers]
        a = np.array(l, dtype=bool)
        c = builtins.sum(l)
        assert_equal(np.count_nonzero(a), c)
        av = a.view(np.uint8)
        av *= 3
        assert_equal(np.count_nonzero(a), c)
        av *= 4
        assert_equal(np.count_nonzero(a), c)
        av[av != 0] = 255
        assert_equal(np.count_nonzero(a), c)