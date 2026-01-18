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
def test_non_array_input(self):
    a = np.require([1, 2, 3, 4], 'i4', ['C', 'A', 'O'])
    assert_(a.flags['O'])
    assert_(a.flags['C'])
    assert_(a.flags['A'])
    assert_(a.dtype == 'i4')
    assert_equal(a, [1, 2, 3, 4])