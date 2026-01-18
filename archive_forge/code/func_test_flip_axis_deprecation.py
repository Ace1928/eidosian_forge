import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
@expires('5.0.0')
def test_flip_axis_deprecation():
    a = np.arange(24).reshape((2, 3, 4))
    axis = 1
    with deprecated_to('5.0.0'):
        a_flipped = flip_axis(a, axis)
    assert_array_equal(a_flipped, np.flip(a, axis))