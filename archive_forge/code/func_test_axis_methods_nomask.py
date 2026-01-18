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
def test_axis_methods_nomask(self):
    a = array([[1, 2, 3], [4, 5, 6]])
    assert_equal(a.sum(0), [5, 7, 9])
    assert_equal(a.sum(-1), [6, 15])
    assert_equal(a.sum(1), [6, 15])
    assert_equal(a.prod(0), [4, 10, 18])
    assert_equal(a.prod(-1), [6, 120])
    assert_equal(a.prod(1), [6, 120])
    assert_equal(a.min(0), [1, 2, 3])
    assert_equal(a.min(-1), [1, 4])
    assert_equal(a.min(1), [1, 4])
    assert_equal(a.max(0), [4, 5, 6])
    assert_equal(a.max(-1), [3, 6])
    assert_equal(a.max(1), [3, 6])