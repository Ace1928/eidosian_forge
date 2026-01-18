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
def test_ip_isclose_allclose(self):
    self._setup()
    tests = self.all_close_tests + self.none_close_tests + self.some_close_tests
    for x, y in tests:
        self.tst_isclose_allclose(x, y)