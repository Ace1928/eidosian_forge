import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal
def test_arithmetic_nans_and_zeros():
    numpy_arr1 = numpy.array([[1, 0, 3], [numpy.nan, 0, numpy.nan]])
    numpy_arr2 = numpy.array([1, 0, 0])
    assert_scalar_or_array_equal(np.array(numpy_arr1) // np.array(numpy_arr2), numpy_arr1 // numpy_arr2)
    assert_scalar_or_array_equal(np.array([0]) // 0, numpy.array([0]) // 0)
    assert_scalar_or_array_equal(np.array([0], dtype=numpy.float64) // 0, numpy.array([0], dtype=numpy.float64) // 0)