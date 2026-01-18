import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_distance_transform_edt5(self):
    out = ndimage.distance_transform_edt(False)
    assert_array_almost_equal(out, [0.0])