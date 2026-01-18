from hypothesis import (
import numpy as np
import pytest
from pandas._libs.byteswap import (
import pandas._testing as tm
@given(read_offset=st.integers(0, 11), number=st.integers(min_value=0))
@example(number=2 ** 16, read_offset=0)
@example(number=2 ** 32, read_offset=0)
@example(number=2 ** 64, read_offset=0)
@pytest.mark.parametrize('int_type', [np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('should_byteswap', [True, False])
def test_int_byteswap(read_offset, number, int_type, should_byteswap):
    assume(number < 2 ** (8 * int_type(0).itemsize))
    _test(number, int_type, read_offset, should_byteswap)