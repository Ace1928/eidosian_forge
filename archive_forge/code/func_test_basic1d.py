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
def test_basic1d(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    assert_(not isMaskedArray(x))
    assert_(isMaskedArray(xm))
    assert_((xm - ym).filled(0).any())
    fail_if_equal(xm.mask.astype(int), ym.mask.astype(int))
    s = x.shape
    assert_equal(np.shape(xm), s)
    assert_equal(xm.shape, s)
    assert_equal(xm.dtype, x.dtype)
    assert_equal(zm.dtype, z.dtype)
    assert_equal(xm.size, reduce(lambda x, y: x * y, s))
    assert_equal(count(xm), len(m1) - reduce(lambda x, y: x + y, m1))
    assert_array_equal(xm, xf)
    assert_array_equal(filled(xm, 1e+20), xf)
    assert_array_equal(x, xm)