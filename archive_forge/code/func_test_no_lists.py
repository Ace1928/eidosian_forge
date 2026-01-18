import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_no_lists(self, block):
    assert_equal(block(1), np.array(1))
    assert_equal(block(np.eye(3)), np.eye(3))