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
def test_axis_keyword(self):
    a3 = np.array([[2, 3], [0, 1], [6, 7], [4, 5]])
    for a in [a3, np.random.randint(0, 100, size=(2, 3, 4))]:
        orig = a.copy()
        np.median(a, axis=None)
        for ax in range(a.ndim):
            np.median(a, axis=ax)
        assert_array_equal(a, orig)
    assert_allclose(np.median(a3, axis=0), [3, 4])
    assert_allclose(np.median(a3.T, axis=1), [3, 4])
    assert_allclose(np.median(a3), 3.5)
    assert_allclose(np.median(a3, axis=None), 3.5)
    assert_allclose(np.median(a3.T), 3.5)