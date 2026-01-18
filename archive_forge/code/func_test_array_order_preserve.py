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
def test_array_order_preserve(self):
    k = np.arange(10).reshape(2, 5, order='F')
    m = delete(k, slice(60, None), axis=1)
    assert_equal(m.flags.c_contiguous, k.flags.c_contiguous)
    assert_equal(m.flags.f_contiguous, k.flags.f_contiguous)