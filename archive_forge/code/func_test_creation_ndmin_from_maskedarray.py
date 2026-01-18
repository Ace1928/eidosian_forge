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
def test_creation_ndmin_from_maskedarray(self):
    x = array([1, 2, 3])
    x[-1] = masked
    xx = array(x, ndmin=2, dtype=float)
    assert_equal(x.shape, x._mask.shape)
    assert_equal(xx.shape, xx._mask.shape)