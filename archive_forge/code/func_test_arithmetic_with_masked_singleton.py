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
def test_arithmetic_with_masked_singleton(self):
    x = masked_array([1, 2])
    y = x * masked
    assert_equal(y.shape, x.shape)
    assert_equal(y._mask, [True, True])
    y = x[0] * masked
    assert_(y is masked)
    y = x + masked
    assert_equal(y.shape, x.shape)
    assert_equal(y._mask, [True, True])