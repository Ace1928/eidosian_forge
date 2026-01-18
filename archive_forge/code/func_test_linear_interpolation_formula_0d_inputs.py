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
def test_linear_interpolation_formula_0d_inputs(self):
    a = np.array(2)
    b = np.array(5)
    t = np.array(0.2)
    assert nfb._lerp(a, b, t) == 2.6