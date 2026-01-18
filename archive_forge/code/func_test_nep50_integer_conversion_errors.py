import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
def test_nep50_integer_conversion_errors():
    np._set_promotion_state('weak')
    with pytest.raises(OverflowError, match='.*uint8'):
        np.array([1], np.uint8) + 300
    with pytest.raises(OverflowError, match='.*uint8'):
        np.uint8(1) + 300
    with pytest.raises(OverflowError, match='Python integer -1 out of bounds for uint8'):
        np.uint8(1) + -1