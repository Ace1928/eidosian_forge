from inspect import getfullargspec
from numpy.testing import assert_raises
from .. import asarray, _elementwise_functions
from .._elementwise_functions import bitwise_left_shift, bitwise_right_shift
from .._dtypes import (
def test_bitwise_shift_error():
    assert_raises(ValueError, lambda: bitwise_left_shift(asarray([1, 1]), asarray([1, -1])))
    assert_raises(ValueError, lambda: bitwise_right_shift(asarray([1, 1]), asarray([1, -1])))