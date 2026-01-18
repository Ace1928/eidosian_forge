import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('fscalar', [np.float16, np.float32])
def test_int_float_promotion_truediv(fscalar):
    i = np.int8(1)
    f = fscalar(1)
    expected = np.result_type(i, f)
    assert (i / f).dtype == expected
    assert (f / i).dtype == expected
    assert (i / i).dtype == np.dtype('float64')
    assert (np.int16(1) / f).dtype == np.dtype('float32')