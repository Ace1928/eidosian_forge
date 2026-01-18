import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import (
from .core import mask_or, getmask, masked_array, nomask, masked, filled
from unittest import TestCase
def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """

    def compare(x, y):
        """Returns the result of the loose comparison between x and y)."""
        return approx(x, y, rtol=10.0 ** (-decimal))
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose, header='Arrays are not almost equal')