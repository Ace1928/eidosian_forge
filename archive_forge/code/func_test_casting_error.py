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
def test_casting_error(self):
    x = [1, 2, 3 + 1j]
    bins = [1, 2, 3]
    assert_raises(TypeError, digitize, x, bins)
    x, bins = (bins, x)
    assert_raises(TypeError, digitize, x, bins)