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
def test_ddof1(self):
    assert_almost_equal(np.var(self.A, ddof=1), self.real_var * len(self.A) / (len(self.A) - 1))
    assert_almost_equal(np.std(self.A, ddof=1) ** 2, self.real_var * len(self.A) / (len(self.A) - 1))