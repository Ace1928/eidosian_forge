import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
@pytest.mark.skipif(not hasattr(ct, 'pythonapi'), reason='`ctypes.pythonapi` required for capsule unpacking.')
def test_loop_access(self):
    data_t = ct.ARRAY(ct.c_char_p, 2)
    dim_t = ct.ARRAY(ct.c_ssize_t, 1)
    strides_t = ct.ARRAY(ct.c_ssize_t, 2)
    strided_loop_t = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, data_t, dim_t, strides_t, ct.c_void_p)

    class call_info_t(ct.Structure):
        _fields_ = [('strided_loop', strided_loop_t), ('context', ct.c_void_p), ('auxdata', ct.c_void_p), ('requires_pyapi', ct.c_byte), ('no_floatingpoint_errors', ct.c_byte)]
    i4 = np.dtype('i4')
    dt, call_info_obj = np.negative._resolve_dtypes_and_context((i4, i4))
    assert dt == (i4, i4)
    np.negative._get_strided_loop(call_info_obj)
    ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
    call_info = ct.pythonapi.PyCapsule_GetPointer(ct.py_object(call_info_obj), ct.c_char_p(b'numpy_1.24_ufunc_call_info'))
    call_info = ct.cast(call_info, ct.POINTER(call_info_t)).contents
    arr = np.arange(10, dtype=i4)
    call_info.strided_loop(call_info.context, data_t(arr.ctypes.data, arr.ctypes.data), arr.ctypes.shape, strides_t(arr.ctypes.strides[0], arr.ctypes.strides[0]), call_info.auxdata)
    assert_array_equal(arr, -np.arange(10, dtype=i4))