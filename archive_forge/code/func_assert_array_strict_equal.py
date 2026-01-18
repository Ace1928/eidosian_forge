import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def assert_array_strict_equal(x, y):
    assert_array_equal(x, y)
    if (x.dtype.alignment <= 8 or np.intp().dtype.itemsize != 4) and sys.platform != 'win32':
        assert_(x.flags == y.flags)
    else:
        assert_(x.flags.owndata == y.flags.owndata)
        assert_(x.flags.writeable == y.flags.writeable)
        assert_(x.flags.c_contiguous == y.flags.c_contiguous)
        assert_(x.flags.f_contiguous == y.flags.f_contiguous)
        assert_(x.flags.writebackifcopy == y.flags.writebackifcopy)
    assert_(x.dtype.isnative == y.dtype.isnative)