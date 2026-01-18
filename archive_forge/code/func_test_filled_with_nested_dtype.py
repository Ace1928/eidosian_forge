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
def test_filled_with_nested_dtype(self):
    ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
    a = array([(1, (1, 1)), (2, (2, 2))], mask=[(0, (1, 0)), (0, (0, 1))], dtype=ndtype)
    test = a.filled(0)
    control = np.array([(1, (0, 1)), (2, (2, 0))], dtype=ndtype)
    assert_equal(test, control)
    test = a['B'].filled(0)
    control = np.array([(0, 1), (2, 0)], dtype=a['B'].dtype)
    assert_equal(test, control)
    Z = numpy.ma.zeros(2, numpy.dtype([('A', '(2,2)i1,(2,2)i1', (2, 2))]))
    assert_equal(Z.data.dtype, numpy.dtype([('A', [('f0', 'i1', (2, 2)), ('f1', 'i1', (2, 2))], (2, 2))]))
    assert_equal(Z.mask.dtype, numpy.dtype([('A', [('f0', '?', (2, 2)), ('f1', '?', (2, 2))], (2, 2))]))