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
def test_minmax_func(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    xr = np.ravel(x)
    xmr = ravel(xm)
    assert_equal(max(xr), maximum.reduce(xmr))
    assert_equal(min(xr), minimum.reduce(xmr))
    assert_equal(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3])
    assert_equal(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9])
    x = arange(5)
    y = arange(5) - 2
    x[3] = masked
    y[0] = masked
    assert_equal(minimum(x, y), where(less(x, y), x, y))
    assert_equal(maximum(x, y), where(greater(x, y), x, y))
    assert_(minimum.reduce(x) == 0)
    assert_(maximum.reduce(x) == 4)
    x = arange(4).reshape(2, 2)
    x[-1, -1] = masked
    assert_equal(maximum.reduce(x, axis=None), 2)