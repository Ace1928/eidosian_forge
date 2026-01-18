import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_integer_0_split(self):
    a = np.arange(10)
    assert_raises(ValueError, array_split, a, 0)