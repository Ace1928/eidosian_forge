import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_invalid_result_type(self):

    def fail():
        _vec_string(['a'], np.int_, 'strip')
    assert_raises(TypeError, fail)