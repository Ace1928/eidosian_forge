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
def test_clip_with_out_simple_int32(self):
    a = self._generate_int32_data(self.nr, self.nc)
    m = np.int32(-1)
    M = np.int32(1)
    ac = np.zeros(a.shape, dtype=np.int64)
    act = ac.copy()
    self.fastclip(a, m, M, ac)
    self.clip(a, m, M, act)
    assert_array_strict_equal(ac, act)