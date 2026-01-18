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
def test_diagonal_view(self):
    x = np.ma.zeros((3, 3))
    x[0, 0] = 10
    x[1, 1] = np.ma.masked
    x[2, 2] = 20
    xd = x.diagonal()
    x[1, 1] = 15
    assert_equal(xd.mask, x.diagonal().mask)
    assert_equal(xd.data, x.diagonal().data)