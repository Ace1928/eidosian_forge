import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
@pytest.mark.parametrize(['float_t', 'uint_t', 'bits'], [(np.float32, np.uint32, 23), (np.float64, np.uint64, 52)])
def test_half_conversion_denormal_round_even(self, float_t, uint_t, bits):
    smallest_value = np.uint16(1).view(np.float16).astype(float_t)
    assert smallest_value == 2 ** (-24)
    rounded_to_zero = smallest_value / float_t(2)
    assert rounded_to_zero.astype(np.float16) == 0
    for i in range(bits):
        larger_pattern = rounded_to_zero.view(uint_t) | uint_t(1 << i)
        larger_value = larger_pattern.view(float_t)
        assert larger_value.astype(np.float16) == smallest_value