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
def test_masked_otherfunctions(self):
    assert_equal(masked_inside(list(range(5)), 1, 3), [0, 199, 199, 199, 4])
    assert_equal(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199])
    assert_equal(masked_inside(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 1, 3).mask, [1, 1, 1, 1, 0])
    assert_equal(masked_outside(array(list(range(5)), mask=[0, 1, 0, 0, 0]), 1, 3).mask, [1, 1, 0, 0, 1])
    assert_equal(masked_equal(array(list(range(5)), mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 0])
    assert_equal(masked_not_equal(array([2, 2, 1, 2, 1], mask=[1, 0, 0, 0, 0]), 2).mask, [1, 0, 1, 0, 1])