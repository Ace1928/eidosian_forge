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
def test_masked_array_no_copy():
    a = np.ma.array([1, 2, 3, 4])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [False, False, True, False])
    a = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 0])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [True, False, True, False])
    a = np.ma.array([np.inf, 1, 2, 3, 4])
    _ = np.ma.masked_invalid(a, copy=False)
    assert_array_equal(a.mask, [True, False, False, False, False])