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
def test_inplace_pow_type(self):
    for t in self.othertypes:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            x = array([1, 2, 3], mask=[0, 0, 1], dtype=t)
            xx = x ** t(2)
            xx_r = array([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
            assert_equal(xx.data, xx_r.data)
            assert_equal(xx.mask, xx_r.mask)
            x **= t(2)
            assert_equal(x.data, xx_r.data)
            assert_equal(x.mask, xx_r.mask)