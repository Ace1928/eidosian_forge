from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
def test_dct(self):
    for dtype in self.real_dtypes:
        self._check_1d(dct, dtype, (16,), -1)
        self._check_1d(dct, dtype, (16, 2), 0)
        self._check_1d(dct, dtype, (2, 16), 1)