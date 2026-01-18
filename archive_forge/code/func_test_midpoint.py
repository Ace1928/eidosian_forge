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
@pytest.mark.parametrize('dtype', TYPE_CODES)
def test_midpoint(self, dtype):
    assert_equal(np.percentile(np.arange(10, dtype=dtype), 51, method='midpoint'), 4.5)
    assert_equal(np.percentile(np.arange(9, dtype=dtype) + 1, 50, method='midpoint'), 5)
    assert_equal(np.percentile(np.arange(11, dtype=dtype), 51, method='midpoint'), 5.5)
    assert_equal(np.percentile(np.arange(11, dtype=dtype), 50, method='midpoint'), 5)