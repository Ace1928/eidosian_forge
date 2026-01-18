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
@pytest.mark.parametrize('dtype', np.typecodes['Float'])
def test_linear_nan_1D(self, dtype):
    arr = np.asarray([15.0, np.NAN, 35.0, 40.0, 50.0], dtype=dtype)
    res = np.percentile(arr, 40.0, method='linear')
    np.testing.assert_equal(res, np.NAN)
    np.testing.assert_equal(res.dtype, arr.dtype)