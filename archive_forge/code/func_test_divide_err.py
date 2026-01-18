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
@pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
@pytest.mark.skipif(platform.machine() == 'armv5tel', reason='See gh-413.')
def test_divide_err(self):
    with np.errstate(divide='raise'):
        with assert_raises(FloatingPointError):
            np.array([1.0]) / np.array([0.0])
        np.seterr(divide='ignore')
        np.array([1.0]) / np.array([0.0])