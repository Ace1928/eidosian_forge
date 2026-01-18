import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_invalid_function_args(self):

    def fail():
        _vec_string(['a'], np.bytes_, 'strip', (1,))
    assert_raises(TypeError, fail)