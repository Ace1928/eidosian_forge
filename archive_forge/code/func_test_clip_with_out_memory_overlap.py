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
def test_clip_with_out_memory_overlap(self):
    a = np.arange(16).reshape(4, 4)
    ac = a.copy()
    a[:-1].clip(4, 10, out=a[1:])
    expected = self.clip(ac[:-1], 4, 10)
    assert_array_equal(a[1:], expected)