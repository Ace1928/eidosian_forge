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
def test_check_on_scalar(self):
    _check_fill_value = np.ma.core._check_fill_value
    fval = _check_fill_value(0, int)
    assert_equal(fval, 0)
    fval = _check_fill_value(None, int)
    assert_equal(fval, default_fill_value(0))
    fval = _check_fill_value(0, '|S3')
    assert_equal(fval, b'0')
    fval = _check_fill_value(None, '|S3')
    assert_equal(fval, default_fill_value(b'camelot!'))
    assert_raises(TypeError, _check_fill_value, 1e+20, int)
    assert_raises(TypeError, _check_fill_value, 'stuff', int)