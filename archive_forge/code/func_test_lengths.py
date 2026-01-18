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
def test_lengths(self):
    expected = np.array(list(self.makegen()))
    a = np.fromiter(self.makegen(), int)
    a20 = np.fromiter(self.makegen(), int, 20)
    assert_(len(a) == len(expected))
    assert_(len(a20) == 20)
    assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)