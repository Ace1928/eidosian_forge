import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_hit_or_miss_input_as_output():
    rstate = numpy.random.RandomState(123)
    data = rstate.randint(low=0, high=2, size=100).astype(bool)
    data_orig = data.copy()
    expected = ndimage.binary_hit_or_miss(data)
    assert_array_equal(data, data_orig)
    ndimage.binary_hit_or_miss(data, output=data)
    assert_array_equal(expected, data)