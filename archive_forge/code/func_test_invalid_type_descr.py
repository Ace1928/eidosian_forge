import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_invalid_type_descr(self):

    def fail():
        _vec_string(['a'], 'BOGUS', 'strip')
    assert_raises(TypeError, fail)