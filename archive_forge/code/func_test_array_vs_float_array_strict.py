import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_vs_float_array_strict(self):
    """Test comparing two arrays with strict option."""
    a = np.array([1, 1, 1])
    b = np.array([1.0, 1.0, 1.0])
    with pytest.raises(AssertionError):
        assert_array_equal(a, b, strict=True)