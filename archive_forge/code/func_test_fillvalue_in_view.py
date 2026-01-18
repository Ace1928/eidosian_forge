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
def test_fillvalue_in_view(self):
    x = array([1, 2, 3], fill_value=1, dtype=np.int64)
    y = x.view()
    assert_(y.fill_value == 1)
    y = x.view(MaskedArray)
    assert_(y.fill_value == 1)
    y = x.view(type=MaskedArray)
    assert_(y.fill_value == 1)
    y = x.view(np.ndarray)
    y = x.view(type=np.ndarray)
    y = x.view(MaskedArray, fill_value=2)
    assert_(y.fill_value == 2)
    y = x.view(type=MaskedArray, fill_value=2)
    assert_(y.fill_value == 2)
    y = x.view(dtype=np.int32)
    assert_(y.fill_value == 999999)