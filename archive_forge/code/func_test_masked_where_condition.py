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
def test_masked_where_condition(self):
    x = array([1.0, 2.0, 3.0, 4.0, 5.0])
    x[2] = masked
    assert_equal(masked_where(greater(x, 2), x), masked_greater(x, 2))
    assert_equal(masked_where(greater_equal(x, 2), x), masked_greater_equal(x, 2))
    assert_equal(masked_where(less(x, 2), x), masked_less(x, 2))
    assert_equal(masked_where(less_equal(x, 2), x), masked_less_equal(x, 2))
    assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
    assert_equal(masked_where(equal(x, 2), x), masked_equal(x, 2))
    assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
    assert_equal(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]), [99, 99, 3, 4, 5])