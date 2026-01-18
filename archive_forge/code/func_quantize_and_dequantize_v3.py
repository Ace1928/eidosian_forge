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
def quantize_and_dequantize_v3(input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], input_min: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], input_max: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], num_bits: _atypes.TensorFuzzingAnnotation[_atypes.Int32], signed_input: bool=True, range_given: bool=True, narrow_range: bool=False, axis: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T]:
    """Quantizes then dequantizes a tensor.

  This is almost identical to QuantizeAndDequantizeV2, except that num_bits is a
  tensor, so its value can change during training.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_min: A `Tensor`. Must have the same type as `input`.
    input_max: A `Tensor`. Must have the same type as `input`.
    num_bits: A `Tensor` of type `int32`.
    signed_input: An optional `bool`. Defaults to `True`.
    range_given: An optional `bool`. Defaults to `True`.
    narrow_range: An optional `bool`. Defaults to `False`.
    axis: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizeAndDequantizeV3', name, input, input_min, input_max, num_bits, 'signed_input', signed_input, 'range_given', range_given, 'narrow_range', narrow_range, 'axis', axis)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantize_and_dequantize_v3_eager_fallback(input, input_min, input_max, num_bits, signed_input=signed_input, range_given=range_given, narrow_range=narrow_range, axis=axis, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if signed_input is None:
        signed_input = True
    signed_input = _execute.make_bool(signed_input, 'signed_input')
    if range_given is None:
        range_given = True
    range_given = _execute.make_bool(range_given, 'range_given')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizeAndDequantizeV3', input=input, input_min=input_min, input_max=input_max, num_bits=num_bits, signed_input=signed_input, range_given=range_given, narrow_range=narrow_range, axis=axis, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('signed_input', _op._get_attr_bool('signed_input'), 'range_given', _op._get_attr_bool('range_given'), 'T', _op._get_attr_type('T'), 'narrow_range', _op._get_attr_bool('narrow_range'), 'axis', _op._get_attr_int('axis'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizeAndDequantizeV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result