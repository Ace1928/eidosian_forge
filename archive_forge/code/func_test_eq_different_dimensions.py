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
def test_eq_different_dimensions(self):
    m1 = array([1, 1], mask=[0, 1])
    for m2 in (array([[0, 1], [1, 2]]), np.array([[0, 1], [1, 2]])):
        test = m1 == m2
        assert_equal(test.data, [[False, False], [True, False]])
        assert_equal(test.mask, [[False, True], [False, True]])