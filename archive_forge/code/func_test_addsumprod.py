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
def test_addsumprod(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    assert_equal(np.add.reduce(x), add.reduce(x))
    assert_equal(np.add.accumulate(x), add.accumulate(x))
    assert_equal(4, sum(array(4), axis=0))
    assert_equal(4, sum(array(4), axis=0))
    assert_equal(np.sum(x, axis=0), sum(x, axis=0))
    assert_equal(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0))
    assert_equal(np.sum(x, 0), sum(x, 0))
    assert_equal(np.prod(x, axis=0), product(x, axis=0))
    assert_equal(np.prod(x, 0), product(x, 0))
    assert_equal(np.prod(filled(xm, 1), axis=0), product(xm, axis=0))
    s = (3, 4)
    x.shape = y.shape = xm.shape = ym.shape = s
    if len(s) > 1:
        assert_equal(np.concatenate((x, y), 1), concatenate((xm, ym), 1))
        assert_equal(np.add.reduce(x, 1), add.reduce(x, 1))
        assert_equal(np.sum(x, 1), sum(x, 1))
        assert_equal(np.prod(x, 1), product(x, 1))