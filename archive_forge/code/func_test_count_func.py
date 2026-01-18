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
def test_count_func(self):
    assert_equal(1, count(1))
    assert_equal(0, array(1, mask=[1]))
    ott = array([0.0, 1.0, 2.0, 3.0], mask=[1, 0, 0, 0])
    res = count(ott)
    assert_(res.dtype.type is np.intp)
    assert_equal(3, res)
    ott = ott.reshape((2, 2))
    res = count(ott)
    assert_(res.dtype.type is np.intp)
    assert_equal(3, res)
    res = count(ott, 0)
    assert_(isinstance(res, ndarray))
    assert_equal([1, 2], res)
    assert_(getmask(res) is nomask)
    ott = array([0.0, 1.0, 2.0, 3.0])
    res = count(ott, 0)
    assert_(isinstance(res, ndarray))
    assert_(res.dtype.type is np.intp)
    assert_raises(np.AxisError, ott.count, axis=1)