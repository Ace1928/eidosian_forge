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
def test_extremum_fill_value(self):
    a = array([(1, (2, 3)), (4, (5, 6))], dtype=[('A', int), ('B', [('BA', int), ('BB', int)])])
    test = a.fill_value
    assert_equal(test.dtype, a.dtype)
    assert_equal(test['A'], default_fill_value(a['A']))
    assert_equal(test['B']['BA'], default_fill_value(a['B']['BA']))
    assert_equal(test['B']['BB'], default_fill_value(a['B']['BB']))
    test = minimum_fill_value(a)
    assert_equal(test.dtype, a.dtype)
    assert_equal(test[0], minimum_fill_value(a['A']))
    assert_equal(test[1][0], minimum_fill_value(a['B']['BA']))
    assert_equal(test[1][1], minimum_fill_value(a['B']['BB']))
    assert_equal(test[1], minimum_fill_value(a['B']))
    test = maximum_fill_value(a)
    assert_equal(test.dtype, a.dtype)
    assert_equal(test[0], maximum_fill_value(a['A']))
    assert_equal(test[1][0], maximum_fill_value(a['B']['BA']))
    assert_equal(test[1][1], maximum_fill_value(a['B']['BB']))
    assert_equal(test[1], maximum_fill_value(a['B']))