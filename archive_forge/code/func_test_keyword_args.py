from scipy.fft._helper import next_fast_len, _init_nd_shape_and_axes
from numpy.testing import assert_equal
from pytest import raises as assert_raises
import pytest
import numpy as np
import sys
from scipy.conftest import (
from scipy._lib._array_api import xp_assert_close, SCIPY_DEVICE
from scipy import fft
def test_keyword_args(self):
    assert next_fast_len(11, real=True) == 12
    assert next_fast_len(target=7, real=False) == 7