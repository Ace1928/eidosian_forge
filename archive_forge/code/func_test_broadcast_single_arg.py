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
def test_broadcast_single_arg(self):
    arrs = [np.empty((5, 6, 7))]
    mit = np.broadcast(*arrs)
    assert_equal(mit.shape, (5, 6, 7))
    assert_equal(mit.ndim, 3)
    assert_equal(mit.nd, 3)
    assert_equal(mit.numiter, 1)
    assert_(arrs[0] is mit.iters[0].base)