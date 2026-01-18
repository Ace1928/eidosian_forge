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
def test_clip_with_out_transposed(self):
    a = np.arange(16).reshape(4, 4)
    out = np.empty_like(a).T
    a.clip(4, 10, out=out)
    expected = self.clip(a, 4, 10)
    assert_array_equal(out, expected)