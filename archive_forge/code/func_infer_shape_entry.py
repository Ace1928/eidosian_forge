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
def infer_shape_entry(num_tensor, tensor_dims, tensor_shapes, _):
    """C Callback for ``CustomOpProp::InferShape``."""
    try:
        n_in = len(op_prop.list_arguments())
        n_out = len(op_prop.list_outputs())
        n_aux = len(op_prop.list_auxiliary_states())
        assert num_tensor == n_in + n_out + n_aux
        shapes = [[tensor_shapes[i][j] for j in range(tensor_dims[i])] for i in range(n_in)]
        ret = op_prop.infer_shape(shapes)
        if len(ret) == 2:
            ishape, oshape = ret
            ashape = []
        elif len(ret) == 3:
            ishape, oshape, ashape = ret
        else:
            raise AssertionError('infer_shape must return 2 or 3 lists')
        assert len(oshape) == n_out, 'InferShape Error: expecting %d entries in returned output shapes, got %d.' % (n_out, len(oshape))
        assert len(ishape) == n_in, 'InferShape Error: expecting %d entries in returned input shapes, got %d.' % (n_in, len(ishape))
        assert len(ashape) == n_aux, 'InferShape Error: expecting %d entries in returned aux state shapes, got %d.' % (n_aux, len(ashape))
        rshape = list(ishape) + list(oshape) + list(ashape)
        for i in range(n_in + n_out + n_aux):
            tensor_shapes[i] = cast(c_array_buf(mx_int, array('i', rshape[i])), POINTER(mx_int))
            tensor_dims[i] = len(rshape[i])
        infer_shape_entry._ref_holder = [tensor_shapes]
    except Exception:
        print('Error in %s.infer_shape: %s' % (reg_name, traceback.format_exc()))
        return False
    return True