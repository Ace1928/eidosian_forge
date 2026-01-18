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
def test_mask_or(self):
    mtype = [('a', bool), ('b', bool)]
    mask = np.array([(0, 0), (0, 1), (1, 0), (0, 0)], dtype=mtype)
    test = mask_or(mask, nomask)
    assert_equal(test, mask)
    test = mask_or(nomask, mask)
    assert_equal(test, mask)
    test = mask_or(mask, False)
    assert_equal(test, mask)
    other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=mtype)
    test = mask_or(mask, other)
    control = np.array([(0, 1), (0, 1), (1, 1), (0, 1)], dtype=mtype)
    assert_equal(test, control)
    othertype = [('A', bool), ('B', bool)]
    other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=othertype)
    try:
        test = mask_or(mask, other)
    except ValueError:
        pass
    dtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
    amask = np.array([(0, (1, 0)), (0, (1, 0))], dtype=dtype)
    bmask = np.array([(1, (0, 1)), (0, (0, 0))], dtype=dtype)
    cntrl = np.array([(1, (1, 1)), (0, (1, 0))], dtype=dtype)
    assert_equal(mask_or(amask, bmask), cntrl)