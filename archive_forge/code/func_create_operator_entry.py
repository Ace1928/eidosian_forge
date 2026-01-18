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
def create_operator_entry(ctx, num_inputs, shapes, ndims, dtypes, ret, _):
    """C Callback for CustomOpProp::CreateOperator"""
    try:
        ctx = py_str(ctx)
        sep = ctx.find('(')
        ctx = context.Context(ctx[:sep], int(ctx[sep + 1:-1]))
        ndims = [ndims[i] for i in range(num_inputs)]
        shapes = [[shapes[i][j] for j in range(ndims[i])] for i in range(num_inputs)]
        dtypes = [dtypes[i] for i in range(num_inputs)]
        op = op_prop.create_operator(ctx, shapes, dtypes)

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

        def backward_entry(num_ndarray, ndarraies, tags, reqs, is_train, _):
            """C Callback for CustomOp::Backward"""
            try:
                tensors = [[] for i in range(5)]
                num_outputs = len(op_prop.list_outputs())
                num_args = len(op_prop.list_arguments())
                for i in range(num_ndarray):
                    if i in _registry.result_deps or i >= num_outputs * 2 + num_args:
                        stype = _STORAGE_TYPE_UNDEFINED
                    else:
                        stype = _STORAGE_TYPE_DEFAULT
                    if tags[i] == 2 or tags[i] == 4:
                        tensors[tags[i]].append(create_ndarray_fn(cast(ndarraies[i], NDArrayHandle), writable=True, stype=stype))
                    else:
                        tensors[tags[i]].append(create_ndarray_fn(cast(ndarraies[i], NDArrayHandle), writable=False, stype=stype))
                reqs = [req_enum[reqs[i]] for i in range(len(tensors[2]))]
                with ctx:
                    op.backward(req=reqs, in_data=tensors[0], out_data=tensors[1], in_grad=tensors[2], out_grad=tensors[3], aux=tensors[4])
            except Exception:
                print('Error in CustomOp.backward: %s' % traceback.format_exc())
                return False
            return True
        cur = _registry.inc()

        def delete_entry(_):
            """C Callback for CustomOp::del"""
            try:
                del _registry.ref_holder[cur]
            except Exception:
                print('Error in CustomOp.delete: %s' % traceback.format_exc())
                return False
            return True
        callbacks = [del_functype(delete_entry), fb_functype(forward_entry), fb_functype(backward_entry)]
        callbacks = [cast(i, CFUNCTYPE(c_int)) for i in callbacks]
        contexts = [None, None, None]
        ret[0] = MXCallbackList(c_int(len(callbacks)), cast(c_array(CFUNCTYPE(c_int), callbacks), POINTER(CFUNCTYPE(c_int))), cast(c_array(c_void_p, contexts), POINTER(c_void_p)))
        op._ref_holder = [ret]
        _registry.ref_holder[cur] = op
    except Exception:
        print('Error in %s.create_operator: %s' % (reg_name, traceback.format_exc()))
        return False
    return True