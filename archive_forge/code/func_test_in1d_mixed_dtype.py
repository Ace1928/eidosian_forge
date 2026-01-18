import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('dtype1,dtype2', [(np.int8, np.int16), (np.int16, np.int8), (np.uint8, np.uint16), (np.uint16, np.uint8), (np.uint8, np.int16), (np.int16, np.uint8)])
@pytest.mark.parametrize('kind', [None, 'sort', 'table'])
def test_in1d_mixed_dtype(self, dtype1, dtype2, kind):
    """Test that in1d works as expected for mixed dtype input."""
    is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
    ar1 = np.array([0, 0, 1, 1], dtype=dtype1)
    if is_dtype2_signed:
        ar2 = np.array([-128, 0, 127], dtype=dtype2)
    else:
        ar2 = np.array([127, 0, 255], dtype=dtype2)
    expected = np.array([True, True, False, False])
    expect_failure = kind == 'table' and any((dtype1 == np.int8 and dtype2 == np.int16, dtype1 == np.int16 and dtype2 == np.int8))
    if expect_failure:
        with pytest.raises(RuntimeError, match='exceed the maximum'):
            in1d(ar1, ar2, kind=kind)
    else:
        assert_array_equal(in1d(ar1, ar2, kind=kind), expected)