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
def test_basicattributes(self):
    a = array([1, 3, 2])
    b = array([1, 3, 2], mask=[1, 0, 1])
    assert_equal(a.ndim, 1)
    assert_equal(b.ndim, 1)
    assert_equal(a.size, 3)
    assert_equal(b.size, 3)
    assert_equal(a.shape, (3,))
    assert_equal(b.shape, (3,))