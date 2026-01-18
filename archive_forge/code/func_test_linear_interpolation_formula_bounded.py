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
@hypothesis.given(t=st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1), a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e+300, max_value=1e+300), b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e+300, max_value=1e+300))
def test_linear_interpolation_formula_bounded(self, t, a, b):
    if a <= b:
        assert a <= nfb._lerp(a, b, t) <= b
    else:
        assert b <= nfb._lerp(a, b, t) <= a