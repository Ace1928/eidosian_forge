from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_asarray_errors():
    assert_raises(TypeError, lambda: Array([1]))
    assert_raises(TypeError, lambda: asarray(['a']))
    assert_raises(ValueError, lambda: asarray([1.0], dtype=np.float16))
    assert_raises(OverflowError, lambda: asarray(2 ** 100))
    assert_raises(TypeError, lambda: asarray([2 ** 100]))
    asarray([1], device='cpu')
    assert_raises(ValueError, lambda: asarray([1], device='gpu'))
    assert_raises(ValueError, lambda: asarray([1], dtype=int))
    assert_raises(ValueError, lambda: asarray([1], dtype='i'))