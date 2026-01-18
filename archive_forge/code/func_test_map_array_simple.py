import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap
from skimage._shared import testing
@pytest.mark.parametrize('dtype_in', _map_array_dtypes_in)
@pytest.mark.parametrize('dtype_out', _map_array_dtypes_out)
@pytest.mark.parametrize('out_array', [True, False])
def test_map_array_simple(dtype_in, dtype_out, out_array):
    input_arr = np.array([0, 2, 0, 3, 4, 5, 0], dtype=dtype_in)
    input_vals = np.array([1, 2, 3, 4, 6], dtype=dtype_in)[::-1]
    output_vals = np.array([6, 7, 8, 9, 10], dtype=dtype_out)[::-1]
    desired = np.array([0, 7, 0, 8, 9, 0, 0], dtype=dtype_out)
    out = None
    if out_array:
        out = np.full(desired.shape, 11, dtype=dtype_out)
    result = map_array(input_arr=input_arr, input_vals=input_vals, output_vals=output_vals, out=out)
    np.testing.assert_array_equal(result, desired)
    assert result.dtype == dtype_out
    if out_array:
        assert out is result