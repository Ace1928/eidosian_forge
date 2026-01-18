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
def test_countnonzero_keepdims(self):
    a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
    assert_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
    assert_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
    assert_equal(np.count_nonzero(a, keepdims=True), [[6]])