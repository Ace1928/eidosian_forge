import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_no_p_overwrite(self):
    p0 = np.array([0, 0.75, 0.25, 0.5, 1.0])
    p = p0.copy()
    np.nanquantile(np.arange(100.0), p, method='midpoint')
    assert_array_equal(p, p0)
    p0 = p0.tolist()
    p = p.tolist()
    np.nanquantile(np.arange(100.0), p, method='midpoint')
    assert_array_equal(p, p0)