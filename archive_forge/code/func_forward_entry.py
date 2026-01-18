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
def forward_entry(num_ndarray, ndarraies, tags, reqs, is_train, _):
    """C Callback for CustomOp::Forward"""
    try:
        tensors = [[] for i in range(5)]
        for i in range(num_ndarray):
            if tags[i] == 1 or tags[i] == 4:
                tensors[tags[i]].append(create_ndarray_fn(cast(ndarraies[i], NDArrayHandle), writable=True))
            else:
                tensors[tags[i]].append(create_ndarray_fn(cast(ndarraies[i], NDArrayHandle), writable=False))
        reqs = [req_enum[reqs[i]] for i in range(len(tensors[1]))]
        with ctx:
            op.forward(is_train=is_train, req=reqs, in_data=tensors[0], out_data=tensors[1], aux=tensors[4])
    except Exception:
        print('Error in CustomOp.forward: %s' % traceback.format_exc())
        return False
    return True