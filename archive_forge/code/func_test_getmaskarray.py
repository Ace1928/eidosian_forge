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
def test_getmaskarray(self):
    ndtype = [('a', int), ('b', float)]
    test = empty(3, dtype=ndtype)
    assert_equal(getmaskarray(test), np.array([(0, 0), (0, 0), (0, 0)], dtype=[('a', '|b1'), ('b', '|b1')]))
    test[:] = masked
    assert_equal(getmaskarray(test), np.array([(1, 1), (1, 1), (1, 1)], dtype=[('a', '|b1'), ('b', '|b1')]))