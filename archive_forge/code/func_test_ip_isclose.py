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
def test_ip_isclose(self):
    self._setup()
    tests = self.some_close_tests
    results = self.some_close_results
    for (x, y), result in zip(tests, results):
        assert_array_equal(np.isclose(x, y), result)