import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b,rtol', [(1.00001, 1.00005, 0.001), (-0.908356 + 0.2j, -0.908358 + 0.2j, 0.001), (0.1 + 1.009j, 0.1 + 1.006j, 0.1), (0.1001 + 2j, 0.1 + 2.001j, 0.01)])
def test_assert_almost_equal_complex_numbers(a, b, rtol):
    _assert_almost_equal_both(a, b, rtol=rtol)
    _assert_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)