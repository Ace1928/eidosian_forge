import platform
import pytest
import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM
@pytest.mark.parametrize('string_dt', ['S', 'U'])
def test_half_conversion_to_string(self, string_dt):
    expected_dt = np.dtype(f'{string_dt}32')
    assert np.promote_types(np.float16, string_dt) == expected_dt
    assert np.promote_types(string_dt, np.float16) == expected_dt
    arr = np.ones(3, dtype=np.float16).astype(string_dt)
    assert arr.dtype == expected_dt