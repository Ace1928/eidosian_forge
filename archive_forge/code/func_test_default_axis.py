import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_default_axis(self):
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[6, 5, 4], [3, 2, 1]])
    assert_equal(np.flip(a), b)