import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.link import (
def test_interval_raises():
    """Test that interval with low > high raises ValueError."""
    with pytest.raises(ValueError, match='One must have low <= high; got low=1, high=0.'):
        Interval(1, 0, False, False)