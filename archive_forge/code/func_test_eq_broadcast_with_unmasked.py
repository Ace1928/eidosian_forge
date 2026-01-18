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
@pytest.mark.parametrize('op', [operator.eq, operator.lt])
def test_eq_broadcast_with_unmasked(self, op):
    a = array([0, 1], mask=[0, 1])
    b = np.arange(10).reshape(5, 2)
    result = op(a, b)
    assert_(result.mask.shape == b.shape)
    assert_equal(result.mask, np.zeros(b.shape, bool) | a.mask)