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
def test_hardmask(self):
    d = arange(5)
    n = [0, 0, 0, 1, 1]
    m = make_mask(n)
    xh = array(d, mask=m, hard_mask=True)
    xs = array(d, mask=m, hard_mask=False, copy=True)
    xh[[1, 4]] = [10, 40]
    xs[[1, 4]] = [10, 40]
    assert_equal(xh._data, [0, 10, 2, 3, 4])
    assert_equal(xs._data, [0, 10, 2, 3, 40])
    assert_equal(xs.mask, [0, 0, 0, 1, 0])
    assert_(xh._hardmask)
    assert_(not xs._hardmask)
    xh[1:4] = [10, 20, 30]
    xs[1:4] = [10, 20, 30]
    assert_equal(xh._data, [0, 10, 20, 3, 4])
    assert_equal(xs._data, [0, 10, 20, 30, 40])
    assert_equal(xs.mask, nomask)
    xh[0] = masked
    xs[0] = masked
    assert_equal(xh.mask, [1, 0, 0, 1, 1])
    assert_equal(xs.mask, [1, 0, 0, 0, 0])
    xh[:] = 1
    xs[:] = 1
    assert_equal(xh._data, [0, 1, 1, 3, 4])
    assert_equal(xs._data, [1, 1, 1, 1, 1])
    assert_equal(xh.mask, [1, 0, 0, 1, 1])
    assert_equal(xs.mask, nomask)
    xh.soften_mask()
    xh[:] = arange(5)
    assert_equal(xh._data, [0, 1, 2, 3, 4])
    assert_equal(xh.mask, nomask)
    xh.harden_mask()
    xh[xh < 3] = masked
    assert_equal(xh._data, [0, 1, 2, 3, 4])
    assert_equal(xh._mask, [1, 1, 1, 0, 0])
    xh[filled(xh > 1, False)] = 5
    assert_equal(xh._data, [0, 1, 2, 5, 5])
    assert_equal(xh._mask, [1, 1, 1, 0, 0])
    xh = array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]], hard_mask=True)
    xh[0] = 0
    assert_equal(xh._data, [[1, 0], [3, 4]])
    assert_equal(xh._mask, [[1, 0], [0, 0]])
    xh[-1, -1] = 5
    assert_equal(xh._data, [[1, 0], [3, 5]])
    assert_equal(xh._mask, [[1, 0], [0, 0]])
    xh[filled(xh < 5, False)] = 2
    assert_equal(xh._data, [[1, 2], [2, 5]])
    assert_equal(xh._mask, [[1, 0], [0, 0]])