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
def declare_backward_dependency_entry(out_grad, in_data, out_data, num_dep, deps, _):
    """C Callback for CustomOpProp::DeclareBacwardDependency"""
    try:
        out_grad = [out_grad[i] for i in range(len(op_prop.list_outputs()))]
        in_data = [in_data[i] for i in range(len(op_prop.list_arguments()))]
        out_data = [out_data[i] for i in range(len(op_prop.list_outputs()))]
        rdeps = op_prop.declare_backward_dependency(out_grad, in_data, out_data)
        num_dep[0] = len(rdeps)
        _registry.result_deps = set()
        for dep in rdeps:
            _registry.result_deps.add(dep)
        rdeps = cast(c_array_buf(c_int, array('i', rdeps)), c_int_p)
        deps[0] = rdeps
        declare_backward_dependency_entry._ref_holder = [deps]
    except Exception:
        tb = traceback.format_exc()
        print('Error in %s.declare_backward_dependency: %s' % (reg_name, tb))
        return False
    return True