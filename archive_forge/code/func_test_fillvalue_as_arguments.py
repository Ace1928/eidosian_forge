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
def test_fillvalue_as_arguments(self):
    a = empty(3, fill_value=999.0)
    assert_equal(a.fill_value, 999.0)
    a = ones(3, fill_value=999.0, dtype=float)
    assert_equal(a.fill_value, 999.0)
    a = zeros(3, fill_value=0.0, dtype=complex)
    assert_equal(a.fill_value, 0.0)
    a = identity(3, fill_value=0.0, dtype=complex)
    assert_equal(a.fill_value, 0.0)