import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import (
from .core import mask_or, getmask, masked_array, nomask, masked, filled
from unittest import TestCase
def fail_if_array_equal(x, y, err_msg='', verbose=True):
    """
    Raises an assertion error if two masked arrays are not equal elementwise.

    """

    def compare(x, y):
        return not np.all(approx(x, y))
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose, header='Arrays are not equal')