import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
def test_nep50_integer_regression():
    np._set_promotion_state('legacy')
    arr = np.array(1)
    assert (arr + 2 ** 63).dtype == np.float64
    assert (arr[()] + 2 ** 63).dtype == np.float64