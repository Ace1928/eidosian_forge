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
def test_dot_shape_mismatch(self):
    x = masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
    y = masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
    z = masked_array([[0, 1], [3, 3]])
    x.dot(y, out=z)
    assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
    assert_almost_equal(z.mask, [[0, 1], [0, 0]])