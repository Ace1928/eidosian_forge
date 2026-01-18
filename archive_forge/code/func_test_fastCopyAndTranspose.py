import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize('a', (np.array(2), np.array([3, 2, 7, 0]), np.arange(6).reshape(2, 3)))
def test_fastCopyAndTranspose(a):
    with pytest.deprecated_call():
        b = np.fastCopyAndTranspose(a)
        assert_equal(b, a.T)
        assert b.flags.owndata