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
def test_assign_dtype(self):
    a = np.zeros(4, dtype='f4,i4')
    m = np.ma.array(a)
    m.dtype = np.dtype('f4')
    repr(m)
    assert_equal(m.dtype, np.dtype('f4'))

    def assign():
        m = np.ma.array(a)
        m.dtype = np.dtype('f8')
    assert_raises(ValueError, assign)
    b = a.view(dtype='f4', type=np.ma.MaskedArray)
    assert_equal(b.dtype, np.dtype('f4'))
    a = np.zeros(4, dtype='f4')
    m = np.ma.array(a)
    m.dtype = np.dtype('f4,i4')
    assert_equal(m.dtype, np.dtype('f4,i4'))
    assert_equal(m._mask, np.ma.nomask)