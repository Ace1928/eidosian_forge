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
def test_fromflex(self):
    a = array([1, 2, 3])
    test = fromflex(a.toflex())
    assert_equal(test, a)
    assert_equal(test.mask, a.mask)
    a = array([1, 2, 3], mask=[0, 0, 1])
    test = fromflex(a.toflex())
    assert_equal(test, a)
    assert_equal(test.mask, a.mask)
    a = array([(1, 1.0), (2, 2.0), (3, 3.0)], mask=[(1, 0), (0, 0), (0, 1)], dtype=[('A', int), ('B', float)])
    test = fromflex(a.toflex())
    assert_equal(test, a)
    assert_equal(test.data, a.data)