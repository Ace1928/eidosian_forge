import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_int_input(self):
    x = 1000 * np.arange(1, 11)
    y = np.arange(1, 11)
    value = barycentric_interpolate(x, y, 1000 * 9.5)
    assert_almost_equal(value, 9.5)