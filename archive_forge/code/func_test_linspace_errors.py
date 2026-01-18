from numpy.testing import assert_raises
import numpy as np
from .. import all
from .._creation_functions import (
from .._dtypes import float32, float64
from .._array_object import Array
def test_linspace_errors():
    linspace(0, 1, 10, device='cpu')
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device='gpu'))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype=float))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype='f'))