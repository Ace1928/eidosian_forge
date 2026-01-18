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
def test_hardmask_oncemore_yay(self):
    a = array([1, 2, 3], mask=[1, 0, 0])
    b = a.harden_mask()
    assert_equal(a, b)
    b[0] = 0
    assert_equal(a, b)
    assert_equal(b, array([1, 2, 3], mask=[1, 0, 0]))
    a = b.soften_mask()
    a[0] = 0
    assert_equal(a, b)
    assert_equal(b, array([0, 2, 3], mask=[0, 0, 0]))