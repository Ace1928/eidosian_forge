import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def check_identityless_reduction(self, a):
    a[...] = 1
    a[1, 0, 0] = 0
    assert_equal(np.minimum.reduce(a, axis=None), 0)
    assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
    assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
    assert_equal(np.minimum.reduce(a, axis=(1, 2)), [1, 0])
    assert_equal(np.minimum.reduce(a, axis=0), [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=1), [[1, 1, 1, 1], [0, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=2), [[1, 1, 1], [0, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=()), a)
    a[...] = 1
    a[0, 1, 0] = 0
    assert_equal(np.minimum.reduce(a, axis=None), 0)
    assert_equal(np.minimum.reduce(a, axis=(0, 1)), [0, 1, 1, 1])
    assert_equal(np.minimum.reduce(a, axis=(0, 2)), [1, 0, 1])
    assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
    assert_equal(np.minimum.reduce(a, axis=0), [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=1), [[0, 1, 1, 1], [1, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=2), [[1, 0, 1], [1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=()), a)
    a[...] = 1
    a[0, 0, 1] = 0
    assert_equal(np.minimum.reduce(a, axis=None), 0)
    assert_equal(np.minimum.reduce(a, axis=(0, 1)), [1, 0, 1, 1])
    assert_equal(np.minimum.reduce(a, axis=(0, 2)), [0, 1, 1])
    assert_equal(np.minimum.reduce(a, axis=(1, 2)), [0, 1])
    assert_equal(np.minimum.reduce(a, axis=0), [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=1), [[1, 0, 1, 1], [1, 1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=2), [[0, 1, 1], [1, 1, 1]])
    assert_equal(np.minimum.reduce(a, axis=()), a)