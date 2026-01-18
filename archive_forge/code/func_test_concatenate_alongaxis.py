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
def test_concatenate_alongaxis(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    s = (3, 4)
    x.shape = y.shape = xm.shape = ym.shape = s
    assert_equal(xm.mask, np.reshape(m1, s))
    assert_equal(ym.mask, np.reshape(m2, s))
    xmym = concatenate((xm, ym), 1)
    assert_equal(np.concatenate((x, y), 1), xmym)
    assert_equal(np.concatenate((xm.mask, ym.mask), 1), xmym._mask)
    x = zeros(2)
    y = array(ones(2), mask=[False, True])
    z = concatenate((x, y))
    assert_array_equal(z, [0, 0, 1, 1])
    assert_array_equal(z.mask, [False, False, False, True])
    z = concatenate((y, x))
    assert_array_equal(z, [1, 1, 0, 0])
    assert_array_equal(z.mask, [False, True, False, False])