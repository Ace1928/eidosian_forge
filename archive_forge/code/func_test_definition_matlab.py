from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
def test_definition_matlab(self):
    dt = np.result_type(np.float32, self.rdt)
    for xr, yr in zip(X, Y):
        x = np.array(xr, dtype=dt)
        y = dct(x, norm='ortho', type=2)
        assert_equal(y.dtype, dt)
        assert_array_almost_equal(y, yr, decimal=self.dec)