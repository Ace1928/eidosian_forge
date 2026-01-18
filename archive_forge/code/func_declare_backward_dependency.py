import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
def declare_backward_dependency(out_grad, in_data, out_data, num_dep, deps, _):
    """C Callback for NDArrayOpProp::DeclareBacwardDependency"""
    try:
        out_grad = [out_grad[i] for i in range(len(self.list_outputs()))]
        in_data = [in_data[i] for i in range(len(self.list_arguments()))]
        out_data = [out_data[i] for i in range(len(self.list_outputs()))]
        rdeps = self.declare_backward_dependency(out_grad, in_data, out_data)
        num_dep[0] = len(rdeps)
        rdeps = cast(c_array_buf(c_int, array('i', rdeps)), c_int_p)
        deps[0] = rdeps
    except Exception:
        print('Error in NDArrayOp.declare_backward_dependency: %s' % traceback.format_exc())
        return False
    return True