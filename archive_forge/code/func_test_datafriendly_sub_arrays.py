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
def test_datafriendly_sub_arrays(self):
    a = array([[1, 1], [3, 3]])
    b = array([1, 1], mask=[0, 0])
    a -= b
    assert_equal(a, [[0, 0], [2, 2]])
    if a.mask is not nomask:
        assert_equal(a.mask, [[0, 0], [0, 0]])
    a = array([[1, 1], [3, 3]])
    b = array([1, 1], mask=[0, 1])
    a -= b
    assert_equal(a, [[0, 0], [2, 2]])
    assert_equal(a.mask, [[0, 1], [0, 1]])