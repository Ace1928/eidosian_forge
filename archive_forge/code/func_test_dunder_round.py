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
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64])
def test_dunder_round(self, dtype):
    s = dtype(1)
    assert_(isinstance(round(s), int))
    assert_(isinstance(round(s, None), int))
    assert_(isinstance(round(s, ndigits=None), int))
    assert_equal(round(s), 1)
    assert_equal(round(s, None), 1)
    assert_equal(round(s, ndigits=None), 1)