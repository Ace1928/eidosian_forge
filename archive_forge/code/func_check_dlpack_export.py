import ctypes
from functools import wraps
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
def check_dlpack_export(arr, expected_arr):
    DLTensor = arr.__dlpack__()
    assert PyCapsule_IsValid(DLTensor, b'dltensor') is True
    result = np.from_dlpack(arr)
    np.testing.assert_array_equal(result, expected_arr, strict=True)
    assert arr.__dlpack_device__() == (1, 0)