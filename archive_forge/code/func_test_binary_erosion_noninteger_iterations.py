import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_erosion_noninteger_iterations():
    data = numpy.ones([1])
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=1.5)