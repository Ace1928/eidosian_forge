import ctypes
from functools import wraps
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
@check_bytes_allocated
@pytest.mark.parametrize(('value_type', 'np_type'), [(pa.uint8(), np.uint8), (pa.uint16(), np.uint16), (pa.uint32(), np.uint32), (pa.uint64(), np.uint64), (pa.int8(), np.int8), (pa.int16(), np.int16), (pa.int32(), np.int32), (pa.int64(), np.int64), (pa.float16(), np.float16), (pa.float32(), np.float32), (pa.float64(), np.float64)])
def test_dlpack(value_type, np_type):
    if Version(np.__version__) < Version('1.24.0'):
        pytest.skip('No dlpack support in numpy versions older than 1.22.0, strict keyword in assert_array_equal added in numpy version 1.24.0')
    expected = np.array([1, 2, 3], dtype=np_type)
    arr = pa.array(expected, type=value_type)
    check_dlpack_export(arr, expected)
    arr_sliced = arr.slice(1, 1)
    expected = np.array([2], dtype=np_type)
    check_dlpack_export(arr_sliced, expected)
    arr_sliced = arr.slice(0, 1)
    expected = np.array([1], dtype=np_type)
    check_dlpack_export(arr_sliced, expected)
    arr_sliced = arr.slice(1)
    expected = np.array([2, 3], dtype=np_type)
    check_dlpack_export(arr_sliced, expected)
    arr_zero = pa.array([], type=value_type)
    expected = np.array([], dtype=np_type)
    check_dlpack_export(arr_zero, expected)