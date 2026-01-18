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
def test_ndarray_mask(self):
    a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
    test = np.sqrt(a)
    control = masked_array([-1, 0, 1, np.sqrt(2), -1], mask=[1, 0, 0, 0, 1])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)
    assert_(not isinstance(test.mask, MaskedArray))