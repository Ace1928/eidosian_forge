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
def test_clip_nan(self):
    d = np.arange(7.0)
    assert_equal(d.clip(min=np.nan), np.nan)
    assert_equal(d.clip(max=np.nan), np.nan)
    assert_equal(d.clip(min=np.nan, max=np.nan), np.nan)
    assert_equal(d.clip(min=-2, max=np.nan), np.nan)
    assert_equal(d.clip(min=np.nan, max=10), np.nan)