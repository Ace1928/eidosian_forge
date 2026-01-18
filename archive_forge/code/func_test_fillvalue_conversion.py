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
def test_fillvalue_conversion(self):
    a = array([b'3', b'4', b'5'])
    a._optinfo.update({'comment': 'updated!'})
    b = array(a, dtype=int)
    assert_equal(b._data, [3, 4, 5])
    assert_equal(b.fill_value, default_fill_value(0))
    b = array(a, dtype=float)
    assert_equal(b._data, [3, 4, 5])
    assert_equal(b.fill_value, default_fill_value(0.0))
    b = a.astype(int)
    assert_equal(b._data, [3, 4, 5])
    assert_equal(b.fill_value, default_fill_value(0))
    assert_equal(b._optinfo['comment'], 'updated!')
    b = a.astype([('a', '|S3')])
    assert_equal(b['a']._data, a._data)
    assert_equal(b['a'].fill_value, a.fill_value)