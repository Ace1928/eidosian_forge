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
def get_operator_arguments(op_name):
    """Given operator name, fetch operator arguments - number of arguments,
    argument names, argument types.

    Parameters
    ----------
    op_name: str
        Handle for the operator

    Returns
    -------
    operator_arguments : OperatorArguments, namedtuple with number of arguments, names and types
    """
    op_handle = OpHandle()
    check_call(_LIB.NNGetOpHandle(c_str(op_name), ctypes.byref(op_handle)))
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()
    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(op_handle, ctypes.byref(real_name), ctypes.byref(desc), ctypes.byref(num_args), ctypes.byref(arg_names), ctypes.byref(arg_types), ctypes.byref(arg_descs), ctypes.byref(key_var_num_args), ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    return OperatorArguments(narg, arg_names, arg_types)