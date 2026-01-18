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
def test_clip_inplace_array(self):
    a = self._generate_data(self.nr, self.nc)
    ac = a.copy()
    m = np.zeros(a.shape)
    M = 1.0
    self.fastclip(a, m, M, a)
    self.clip(a, m, M, ac)
    assert_array_strict_equal(a, ac)