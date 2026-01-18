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
def test_check_on_fields(self):
    _check_fill_value = np.ma.core._check_fill_value
    ndtype = [('a', int), ('b', float), ('c', '|S3')]
    fval = _check_fill_value([-999, -12345678.9, '???'], ndtype)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), [-999, -12345678.9, b'???'])
    fval = _check_fill_value(None, ndtype)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), [default_fill_value(0), default_fill_value(0.0), asbytes(default_fill_value('0'))])
    fill_val = np.array((-999, -12345678.9, '???'), dtype=ndtype)
    fval = _check_fill_value(fill_val, ndtype)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), [-999, -12345678.9, b'???'])
    fill_val = np.array((-999, -12345678.9, '???'), dtype=[('A', int), ('B', float), ('C', '|S3')])
    fval = _check_fill_value(fill_val, ndtype)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), [-999, -12345678.9, b'???'])
    fill_val = np.ndarray(shape=(1,), dtype=object)
    fill_val[0] = (-999, -12345678.9, b'???')
    fval = _check_fill_value(fill_val, object)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), [-999, -12345678.9, b'???'])
    ndtype = [('a', int)]
    fval = _check_fill_value(-999999999, ndtype)
    assert_(isinstance(fval, ndarray))
    assert_equal(fval.item(), (-999999999,))