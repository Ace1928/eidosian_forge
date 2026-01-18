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
def test_logical_and_or_xor(self):
    assert_array_equal(self.t | self.t, self.t)
    assert_array_equal(self.f | self.f, self.f)
    assert_array_equal(self.t | self.f, self.t)
    assert_array_equal(self.f | self.t, self.t)
    np.logical_or(self.t, self.t, out=self.o)
    assert_array_equal(self.o, self.t)
    assert_array_equal(self.t & self.t, self.t)
    assert_array_equal(self.f & self.f, self.f)
    assert_array_equal(self.t & self.f, self.f)
    assert_array_equal(self.f & self.t, self.f)
    np.logical_and(self.t, self.t, out=self.o)
    assert_array_equal(self.o, self.t)
    assert_array_equal(self.t ^ self.t, self.f)
    assert_array_equal(self.f ^ self.f, self.f)
    assert_array_equal(self.t ^ self.f, self.t)
    assert_array_equal(self.f ^ self.t, self.t)
    np.logical_xor(self.t, self.t, out=self.o)
    assert_array_equal(self.o, self.f)
    assert_array_equal(self.nm & self.t, self.nm)
    assert_array_equal(self.im & self.f, False)
    assert_array_equal(self.nm & True, self.nm)
    assert_array_equal(self.im & False, self.f)
    assert_array_equal(self.nm | self.t, self.t)
    assert_array_equal(self.im | self.f, self.im)
    assert_array_equal(self.nm | True, self.t)
    assert_array_equal(self.im | False, self.im)
    assert_array_equal(self.nm ^ self.t, self.im)
    assert_array_equal(self.im ^ self.f, self.im)
    assert_array_equal(self.nm ^ True, self.im)
    assert_array_equal(self.im ^ False, self.im)