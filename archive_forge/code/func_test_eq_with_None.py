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
def test_eq_with_None(self):
    with suppress_warnings() as sup:
        sup.filter(FutureWarning, 'Comparison to `None`')
        a = array([None, 1], mask=[0, 1])
        assert_equal(a == None, array([True, False], mask=[0, 1]))
        assert_equal(a.data == None, [True, False])
        assert_equal(a != None, array([False, True], mask=[0, 1]))
        a = array([None, 1], mask=False)
        assert_equal(a == None, [True, False])
        assert_equal(a != None, [False, True])
        a = array([None, 2], mask=True)
        assert_equal(a == None, array([False, True], mask=True))
        assert_equal(a != None, array([True, False], mask=True))
        a = masked
        assert_equal(a == None, masked)