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
def test_fillvalue_individual_fields(self):
    ndtype = [('a', int), ('b', int)]
    a = array(list(zip([1, 2, 3], [4, 5, 6])), fill_value=(-999, -999), dtype=ndtype)
    aa = a['a']
    aa.set_fill_value(10)
    assert_equal(aa._fill_value, np.array(10))
    assert_equal(tuple(a.fill_value), (10, -999))
    a.fill_value['b'] = -10
    assert_equal(tuple(a.fill_value), (10, -10))
    t = array(list(zip([1, 2, 3], [4, 5, 6])), dtype=ndtype)
    tt = t['a']
    tt.set_fill_value(10)
    assert_equal(tt._fill_value, np.array(10))
    assert_equal(tuple(t.fill_value), (10, default_fill_value(0)))