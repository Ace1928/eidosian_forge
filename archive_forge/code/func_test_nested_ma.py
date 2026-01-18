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
def test_nested_ma(self):
    arr = np.ma.array([None, None])
    arr[0, ...] = np.array([np.ma.masked], object)[0, ...]
    assert_(arr.data[0] is np.ma.masked)
    assert_(arr[0] is np.ma.masked)
    arr[0] = np.ma.masked
    assert_(arr[0] is np.ma.masked)