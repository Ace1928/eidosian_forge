import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def params_0(f):
    val = f()
    assert_(np.isscalar(val))
    val = f(10)
    assert_(val.shape == (10,))
    val = f((10, 10))
    assert_(val.shape == (10, 10))
    val = f((10, 10, 10))
    assert_(val.shape == (10, 10, 10))
    val = f(size=(5, 5))
    assert_(val.shape == (5, 5))