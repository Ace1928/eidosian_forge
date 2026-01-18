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
def test_creation_ndmin(self):
    x = array([1, 2, 3], mask=[1, 0, 0], ndmin=2)
    assert_equal(x.shape, (1, 3))
    assert_equal(x._data, [[1, 2, 3]])
    assert_equal(x._mask, [[1, 0, 0]])