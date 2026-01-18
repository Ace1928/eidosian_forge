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
def infer_storage_type_backward_entry(num_tensor, tensor_stypes, tags, _):
    """C Callback for CustomOpProp::InferStorageTypeBackward"""
    try:
        tensors = [[] for i in range(5)]
        for i in range(num_tensor):
            tensors[tags[i]].append(_STORAGE_TYPE_ID_TO_STR[tensor_stypes[i]])
        tensors = [tensors[3], tensors[0], tensors[1], tensors[2], tensors[4]]
        ret = op_prop.infer_storage_type_backward(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4])
        if len(ret) == 4:
            ret += []
        elif len(ret) == 5:
            pass
        else:
            raise AssertionError('infer_storage_type_backward must return 4 or 5 lists')
        assert len(ret[0]) == len(tensors[0]), 'InferStorageTypeBackward Error: expecting == %d entries in returned output gradient stypes, got %d.' % (len(tensors[0]), len(ret[0]))
        assert len(ret[1]) == len(tensors[1]), 'InferStorageTypeBackward Error: expecting == %d entries in returned input stypes, got %d.' % (len(tensors[1]), len(ret[1]))
        assert len(ret[2]) == len(tensors[2]), 'InferStorageTypeBackward Error: expecting == %d entries in returned output stypes, got %d.' % (len(tensors[2]), len(ret[2]))
        assert len(ret[3]) == len(tensors[3]), 'InferStorageTypeBackward Error: expecting == %d entries in returned input gradient stypes, got %d.' % (len(tensors[3]), len(ret[3]))
        assert len(ret[4]) == len(tensors[4]), 'InferStorageTypeBackward Error: expecting == %d entries in returned aux stypes, got %d.' % (len(tensors[4]), len(ret[4]))
        rstype = []
        for i, ret_list in enumerate(ret):
            rstype.extend(ret_list)
        for i, stype in enumerate(rstype):
            assert stype != _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_UNDEFINED], 'stype should not be undefined'
            assert stype in _STORAGE_TYPE_STR_TO_ID, 'Provided stype: %s is not valid valid stypes are %s, %s, %s' % (stype, _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_DEFAULT], _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_ROW_SPARSE], _STORAGE_TYPE_ID_TO_STR[_STORAGE_TYPE_CSR])
            tensor_stypes[i] = _STORAGE_TYPE_STR_TO_ID[stype]
        infer_storage_type_backward_entry._ref_holder = [tensor_stypes]
    except Exception:
        print('Error in %s.infer_type: %s' % (reg_name, traceback.format_exc()))
        return False
    return True