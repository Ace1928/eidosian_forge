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
def test_inplace_division_misc(self):
    x = [1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0]
    y = [5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0]
    m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
    xm = masked_array(x, mask=m1)
    ym = masked_array(y, mask=m2)
    z = xm / ym
    assert_equal(z._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    assert_equal(z._data, [1.0, 1.0, 1.0, -1.0, -pi / 2.0, 4.0, 5.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    xm = xm.copy()
    xm /= ym
    assert_equal(xm._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    assert_equal(z._data, [1.0, 1.0, 1.0, -1.0, -pi / 2.0, 4.0, 5.0, 1.0, 1.0, 1.0, 2.0, 3.0])