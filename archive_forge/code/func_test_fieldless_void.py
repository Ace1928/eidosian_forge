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
def test_fieldless_void():
    dt = np.dtype([])
    x = np.empty(4, dt)
    mx = np.ma.array(x)
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)
    mx = np.ma.array(x, mask=x)
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)