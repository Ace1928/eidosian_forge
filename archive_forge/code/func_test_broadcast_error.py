import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_broadcast_error(self):

    def fail():
        _vec_string([['abc', 'def']], np.int_, 'find', (['a', 'd', 'j'],))
    assert_raises(ValueError, fail)