import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_dilation_square_structure(self):
    result = ndimage.grey_dilation(self.array, structure=self.sq3x3)
    assert_array_almost_equal(result, self.dilated3x3 + 1)