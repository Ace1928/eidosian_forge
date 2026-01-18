import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_bad_legacy_ufunc_silent_errors():
    arr = np.arange(3).astype(np.float64)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error(arr, arr)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        non_contig = arr.repeat(20).reshape(-1, 6)[:, ::2]
        ncu_tests.always_error(non_contig, arr)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error.outer(arr, arr)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error.reduce(arr)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error.reduceat(arr, [0, 1])
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error.accumulate(arr)
    with pytest.raises(RuntimeError, match='How unexpected :\\)!'):
        ncu_tests.always_error.at(arr, [0, 1, 2], arr)