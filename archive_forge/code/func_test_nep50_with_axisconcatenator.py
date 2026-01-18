import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
def test_nep50_with_axisconcatenator():
    np._set_promotion_state('weak')
    with pytest.raises(OverflowError):
        np.r_[np.arange(5, dtype=np.int8), 255]