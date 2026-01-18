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
@pytest.mark.parametrize('dt', ['S', 'U'])
@pytest.mark.parametrize('fill', [None, 'A'])
def test_ne_for_strings(self, dt, fill):
    a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)
    test = a != a
    assert_equal(test.data, [False, False])
    assert_equal(test.mask, [False, True])
    assert_(test.fill_value == True)
    test = a != a[0]
    assert_equal(test.data, [False, True])
    assert_equal(test.mask, [False, True])
    assert_(test.fill_value == True)
    b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)
    test = a != b
    assert_equal(test.data, [True, True])
    assert_equal(test.mask, [True, True])
    assert_(test.fill_value == True)
    test = a[0] != b
    assert_equal(test.data, [True, True])
    assert_equal(test.mask, [True, False])
    assert_(test.fill_value == True)
    test = b != a[0]
    assert_equal(test.data, [True, True])
    assert_equal(test.mask, [True, False])
    assert_(test.fill_value == True)