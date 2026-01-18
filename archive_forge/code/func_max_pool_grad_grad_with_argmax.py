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
def max_pool_grad_grad_with_argmax(input: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradGradWithArgmax_T], grad: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradGradWithArgmax_T], argmax: _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradGradWithArgmax_Targmax], ksize, strides, padding: str, include_batch_in_index: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MaxPoolGradGradWithArgmax_T]:
    """Computes second-order gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      input of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MaxPoolGradGradWithArgmax', name, input, grad, argmax, 'ksize', ksize, 'strides', strides, 'padding', padding, 'include_batch_in_index', include_batch_in_index)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return max_pool_grad_grad_with_argmax_eager_fallback(input, grad, argmax, ksize=ksize, strides=strides, padding=padding, include_batch_in_index=include_batch_in_index, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'max_pool_grad_grad_with_argmax' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'max_pool_grad_grad_with_argmax' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if include_batch_in_index is None:
        include_batch_in_index = False
    include_batch_in_index = _execute.make_bool(include_batch_in_index, 'include_batch_in_index')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MaxPoolGradGradWithArgmax', input=input, grad=grad, argmax=argmax, ksize=ksize, strides=strides, padding=padding, include_batch_in_index=include_batch_in_index, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('ksize', _op.get_attr('ksize'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'include_batch_in_index', _op._get_attr_bool('include_batch_in_index'), 'Targmax', _op._get_attr_type('Targmax'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MaxPoolGradGradWithArgmax', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result