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
def test_masked_where_structured(self):
    a = np.zeros(10, dtype=[('A', '<f2'), ('B', '<f4')])
    with np.errstate(over='ignore'):
        am = np.ma.masked_where(a['A'] < 5, a)
    assert_equal(am.mask.dtype.names, am.dtype.names)
    assert_equal(am['A'], np.ma.masked_array(np.zeros(10), np.ones(10)))