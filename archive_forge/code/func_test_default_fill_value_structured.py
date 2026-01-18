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
def test_default_fill_value_structured(self):
    fields = array([(1, 1, 1)], dtype=[('i', int), ('s', '|S8'), ('f', float)])
    f1 = default_fill_value(fields)
    f2 = default_fill_value(fields.dtype)
    expected = np.array((default_fill_value(0), default_fill_value('0'), default_fill_value(0.0)), dtype=fields.dtype)
    assert_equal(f1, expected)
    assert_equal(f2, expected)