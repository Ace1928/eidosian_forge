import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
def test_bit_count(self):
    for exp in [10, 17, 63]:
        a = 2 ** exp
        assert np.uint64(a).bit_count() == 1
        assert np.uint64(a - 1).bit_count() == exp
        assert np.uint64(a ^ 63).bit_count() == 7
        assert np.uint64(a - 1 ^ 510).bit_count() == exp - 8