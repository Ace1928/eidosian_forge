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
def test_non_array(self):
    a = np.arange(4)

    class array_like:
        __array_interface__ = a.__array_interface__

        def __array_wrap__(self, arr):
            return self
    assert isinstance(np.abs(array_like()), array_like)
    exp = np.i0(a)
    res = np.i0(array_like())
    assert_array_equal(exp, res)