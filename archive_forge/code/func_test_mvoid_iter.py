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
def test_mvoid_iter(self):
    ndtype = [('a', int), ('b', int)]
    a = masked_array([(1, 2), (3, 4)], mask=[(0, 0), (1, 0)], dtype=ndtype)
    assert_equal(list(a[0]), [1, 2])
    assert_equal(list(a[1]), [masked, 4])