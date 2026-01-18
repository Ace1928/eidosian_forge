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
def test_choose_with_out(self):
    choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
    store = empty(4, dtype=int)
    chosen = choose([2, 3, 1, 0], choices, out=store)
    assert_equal(store, array([20, 31, 12, 3]))
    assert_(store is chosen)
    store = empty(4, dtype=int)
    indices_ = array([2, 3, 1, 0], mask=[1, 0, 0, 1])
    chosen = choose(indices_, choices, mode='wrap', out=store)
    assert_equal(store, array([99, 31, 12, 99]))
    assert_equal(store.mask, [1, 0, 0, 1])
    choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
    indices_ = [2, 3, 1, 0]
    store = empty(4, dtype=int).view(ndarray)
    chosen = choose(indices_, choices, mode='wrap', out=store)
    assert_equal(store, array([999999, 31, 12, 999999]))