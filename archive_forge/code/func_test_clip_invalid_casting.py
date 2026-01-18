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
def test_clip_invalid_casting(self):
    a = np.arange(10, dtype=object)
    with assert_raises_regex(ValueError, 'casting must be one of'):
        self.fastclip(a, 1, 8, casting='garbage')