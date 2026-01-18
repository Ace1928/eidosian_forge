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
@pytest.mark.parametrize('dtype', ['m8[s]'])
@pytest.mark.parametrize('pos', [0, 23, 10])
def test_nat_behavior(self, dtype, pos):
    a = np.arange(0, 24, dtype=dtype)
    a[pos] = 'NaT'
    res = np.median(a)
    assert res.dtype == dtype
    assert np.isnat(res)
    res = np.percentile(a, [30, 60])
    assert res.dtype == dtype
    assert np.isnat(res).all()
    a = np.arange(0, 24 * 3, dtype=dtype).reshape(-1, 3)
    a[pos, 1] = 'NaT'
    res = np.median(a, axis=0)
    assert_array_equal(np.isnat(res), [False, True, False])