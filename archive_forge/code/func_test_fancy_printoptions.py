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
def test_fancy_printoptions(self):
    fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
    test = array([(1, (2, 3.0)), (4, (5, 6.0))], mask=[(1, (0, 1)), (0, (1, 0))], dtype=fancydtype)
    control = '[(--, (2, --)) (4, (--, 6.0))]'
    assert_equal(str(test), control)
    t_2d0 = masked_array(data=(0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 0.0), mask=(False, [[True, False, True], [False, False, True]], False), dtype='int, (2,3)float, float')
    control = '(0, [[--, 0.0, --], [0.0, 0.0, --]], 0.0)'
    assert_equal(str(t_2d0), control)