import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_extended_axis_invalid(self):
    d = np.ones((3, 5, 7, 11))
    assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=-5)
    assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=(0, -5))
    assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=4)
    assert_raises(np.AxisError, np.nanpercentile, d, q=5, axis=(0, 4))
    assert_raises(ValueError, np.nanpercentile, d, q=5, axis=(1, 1))