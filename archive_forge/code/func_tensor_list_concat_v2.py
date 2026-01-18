import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def tensor_list_concat_v2(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], element_shape: _atypes.TensorFuzzingAnnotation[TV_TensorListConcatV2_shape_type], leading_dims: _atypes.TensorFuzzingAnnotation[_atypes.Int64], element_dtype: TV_TensorListConcatV2_element_dtype, name=None):
    """Concats all tensors in the list along the 0th dimension.

  Requires that all tensors have the same shape except the first dimension.

  input_handle: The input list.
  element_shape: The shape of the uninitialized elements in the list. If the first
    dimension is not -1, it is assumed that all list elements have the same
    leading dim.
  leading_dims: The list of leading dims of uninitialized list elements. Used if
    the leading dim of input_handle.element_shape or the element_shape input arg
    is not already set.
  tensor: The concated result.
  lengths: Output tensor containing sizes of the 0th dimension of tensors in the list, used for computing the gradient.

  Args:
    input_handle: A `Tensor` of type `variant`.
    element_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    leading_dims: A `Tensor` of type `int64`.
    element_dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (tensor, lengths).

    tensor: A `Tensor` of type `element_dtype`.
    lengths: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListConcatV2', name, input_handle, element_shape, leading_dims, 'element_dtype', element_dtype)
            _result = _TensorListConcatV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_concat_v2_eager_fallback(input_handle, element_shape, leading_dims, element_dtype=element_dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    element_dtype = _execute.make_type(element_dtype, 'element_dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListConcatV2', input_handle=input_handle, element_shape=element_shape, leading_dims=leading_dims, element_dtype=element_dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'), 'shape_type', _op._get_attr_type('shape_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListConcatV2', _inputs_flat, _attrs, _result)
    _result = _TensorListConcatV2Output._make(_result)
    return _result