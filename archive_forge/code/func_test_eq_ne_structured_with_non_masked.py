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
def test_eq_ne_structured_with_non_masked(self):
    a = array([(1, 1), (2, 2), (3, 4)], mask=[(0, 1), (0, 0), (1, 1)], dtype='i4,i4')
    eq = a == a.data
    ne = a.data != a
    assert_(np.all(eq))
    assert_(not np.any(ne))
    expected_mask = a.mask == np.ones((), a.mask.dtype)
    assert_array_equal(eq.mask, expected_mask)
    assert_array_equal(ne.mask, expected_mask)
    assert_equal(eq.data, [True, True, False])
    assert_array_equal(eq.data, ~ne.data)