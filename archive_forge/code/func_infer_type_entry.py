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
def infer_type_entry(num_tensor, tensor_types, _):
    """C Callback for CustomOpProp::InferType"""
    try:
        n_in = len(op_prop.list_arguments())
        n_out = len(op_prop.list_outputs())
        n_aux = len(op_prop.list_auxiliary_states())
        assert num_tensor == n_in + n_out + n_aux
        types = [_DTYPE_MX_TO_NP[tensor_types[i]] for i in range(n_in)]
        ret = op_prop.infer_type(types)
        if len(ret) == 2:
            itype, otype = ret
            atype = []
        elif len(ret) == 3:
            itype, otype, atype = ret
        else:
            raise AssertionError('infer_type must return 2 or 3 lists')
        assert len(otype) == n_out, 'InferType Error: expecting %d entries in returned output types, got %d.' % (n_out, len(otype))
        assert len(itype) == n_in, 'InferType Error: expecting %d entries in returned input types, got %d.' % (n_in, len(itype))
        assert len(atype) == n_aux, 'InferType Error: expecting %d entries in returned aux state types, got %d.' % (n_aux, len(atype))
        rtype = list(itype) + list(otype) + list(atype)
        for i, dtype in enumerate(rtype):
            tensor_types[i] = _DTYPE_NP_TO_MX[dtype]
        infer_type_entry._ref_holder = [tensor_types]
    except Exception:
        print('Error in %s.infer_type: %s' % (reg_name, traceback.format_exc()))
        return False
    return True