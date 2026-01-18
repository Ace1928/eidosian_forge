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
def test_append_masked_array_along_axis():
    a = np.ma.masked_equal([1, 2, 3], value=2)
    b = np.ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)
    assert_raises(ValueError, np.ma.append, a, b, axis=0)
    result = np.ma.append(a[np.newaxis, :], b, axis=0)
    expected = np.ma.arange(1, 10)
    expected[[1, 6]] = np.ma.masked
    expected = expected.reshape((3, 3))
    assert_array_equal(result.data, expected.data)
    assert_array_equal(result.mask, expected.mask)