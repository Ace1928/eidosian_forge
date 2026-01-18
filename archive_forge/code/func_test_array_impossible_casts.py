import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('array', [True, False])
def test_array_impossible_casts(array):
    rt = rational(1, 2)
    if array:
        rt = np.array(rt)
    with assert_raises(TypeError):
        np.array(rt, dtype='M8')