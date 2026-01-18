import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_daub(self):
    with pytest.deprecated_call():
        for i in range(1, 15):
            assert_equal(len(wavelets.daub(i)), i * 2)