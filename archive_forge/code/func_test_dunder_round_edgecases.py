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
@pytest.mark.parametrize('val, ndigits', [pytest.param(2 ** 31 - 1, -1, marks=pytest.mark.xfail(reason='Out of range of int32')), (2 ** 31 - 1, 1 - math.ceil(math.log10(2 ** 31 - 1))), (2 ** 31 - 1, -math.ceil(math.log10(2 ** 31 - 1)))])
def test_dunder_round_edgecases(self, val, ndigits):
    assert_equal(round(val, ndigits), round(np.int32(val), ndigits))