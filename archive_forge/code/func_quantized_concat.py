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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.quantized_concat', v1=['quantization.quantized_concat', 'quantized_concat'])
@deprecated_endpoints('quantized_concat')
def quantized_concat(concat_dim: _atypes.TensorFuzzingAnnotation[_atypes.Int32], values: List[_atypes.TensorFuzzingAnnotation[TV_QuantizedConcat_T]], input_mins: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], input_maxes: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name=None):
    """Concatenates quantized tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects with the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    input_mins: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The minimum scalar values for each of the input tensors.
    input_maxes: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The maximum scalar values for each of the input tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `values`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedConcat', name, concat_dim, values, input_mins, input_maxes)
            _result = _QuantizedConcatOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_quantized_concat((concat_dim, values, input_mins, input_maxes, name), None)
            if _result is not NotImplemented:
                return _result
            return quantized_concat_eager_fallback(concat_dim, values, input_mins, input_maxes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(quantized_concat, (), dict(concat_dim=concat_dim, values=values, input_mins=input_mins, input_maxes=input_maxes, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_quantized_concat((concat_dim, values, input_mins, input_maxes, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'quantized_concat' Op, not %r." % values)
    _attr_N = len(values)
    if not isinstance(input_mins, (list, tuple)):
        raise TypeError("Expected list for 'input_mins' argument to 'quantized_concat' Op, not %r." % input_mins)
    if len(input_mins) != _attr_N:
        raise ValueError("List argument 'input_mins' to 'quantized_concat' Op with length %d must match length %d of argument 'values'." % (len(input_mins), _attr_N))
    if not isinstance(input_maxes, (list, tuple)):
        raise TypeError("Expected list for 'input_maxes' argument to 'quantized_concat' Op, not %r." % input_maxes)
    if len(input_maxes) != _attr_N:
        raise ValueError("List argument 'input_maxes' to 'quantized_concat' Op with length %d must match length %d of argument 'values'." % (len(input_maxes), _attr_N))
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedConcat', concat_dim=concat_dim, values=values, input_mins=input_mins, input_maxes=input_maxes, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(quantized_concat, (), dict(concat_dim=concat_dim, values=values, input_mins=input_mins, input_maxes=input_maxes, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedConcat', _inputs_flat, _attrs, _result)
    _result = _QuantizedConcatOutput._make(_result)
    return _result