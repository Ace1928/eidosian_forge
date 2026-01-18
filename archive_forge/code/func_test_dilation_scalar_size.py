import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_dilation_scalar_size(self):
    result = ndimage.grey_dilation(self.array, size=3)
    assert_array_almost_equal(result, self.dilated3x3)