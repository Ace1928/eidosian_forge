import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_large_concatenate_axis_None(self):
    x = np.arange(1, 100)
    r = np.concatenate(x, None)
    assert_array_equal(x, r)
    r = np.concatenate(x, 100)
    assert_array_equal(x, r)