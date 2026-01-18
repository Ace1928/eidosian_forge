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
def test_append_masked_array():
    a = np.ma.masked_equal([1, 2, 3], value=2)
    b = np.ma.masked_equal([4, 3, 2], value=2)
    result = np.ma.append(a, b)
    expected_data = [1, 2, 3, 4, 3, 2]
    expected_mask = [False, True, False, False, False, True]
    assert_array_equal(result.data, expected_data)
    assert_array_equal(result.mask, expected_mask)
    a = np.ma.masked_all((2, 2))
    b = np.ma.ones((3, 1))
    result = np.ma.append(a, b)
    expected_data = [1] * 3
    expected_mask = [True] * 4 + [False] * 3
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)
    result = np.ma.append(a, b, axis=None)
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)