import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_flatten_structured_array(self):
    ndtype = [('a', int), ('b', float)]
    a = np.array([(1, 1), (2, 2)], dtype=ndtype)
    test = flatten_structured_array(a)
    control = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    assert_equal(test, control)
    assert_equal(test.dtype, control.dtype)
    a = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
    test = flatten_structured_array(a)
    control = array([[1.0, 1.0], [2.0, 2.0]], mask=[[0, 1], [1, 0]], dtype=float)
    assert_equal(test, control)
    assert_equal(test.dtype, control.dtype)
    assert_equal(test.mask, control.mask)
    ndtype = [('a', int), ('b', [('ba', int), ('bb', float)])]
    a = array([(1, (1, 1.1)), (2, (2, 2.2))], mask=[(0, (1, 0)), (1, (0, 1))], dtype=ndtype)
    test = flatten_structured_array(a)
    control = array([[1.0, 1.0, 1.1], [2.0, 2.0, 2.2]], mask=[[0, 1, 0], [1, 0, 1]], dtype=float)
    assert_equal(test, control)
    assert_equal(test.dtype, control.dtype)
    assert_equal(test.mask, control.mask)
    ndtype = [('a', int), ('b', float)]
    a = np.array([[(1, 1)], [(2, 2)]], dtype=ndtype)
    test = flatten_structured_array(a)
    control = np.array([[[1.0, 1.0]], [[2.0, 2.0]]], dtype=float)
    assert_equal(test, control)
    assert_equal(test.dtype, control.dtype)