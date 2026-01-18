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
def test_ip_allclose(self):
    arr = np.array([100, 1000])
    aran = np.arange(125).reshape((5, 5, 5))
    atol = self.atol
    rtol = self.rtol
    data = [([1, 0], [1, 0]), ([atol], [0]), ([1], [1 + rtol + atol]), (arr, arr + arr * rtol), (arr, arr + arr * rtol + atol * 2), (aran, aran + aran * rtol), (np.inf, np.inf), (np.inf, [np.inf])]
    for x, y in data:
        self.tst_allclose(x, y)