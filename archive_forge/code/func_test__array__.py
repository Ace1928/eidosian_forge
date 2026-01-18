import warnings
import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test__array__():
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, 6]])
    modin_arr = np.array(numpy_arr)
    converted_array = numpy.array(modin_arr)
    assert type(converted_array) is type(numpy_arr)
    assert_scalar_or_array_equal(converted_array, numpy_arr)