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
def quantize_and_dequantize_v3_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], input_min: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], input_max: _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T], num_bits: _atypes.TensorFuzzingAnnotation[_atypes.Int32], signed_input: bool, range_given: bool, narrow_range: bool, axis: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_QuantizeAndDequantizeV3_T]:
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
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_min, input_max], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    input, input_min, input_max = _inputs_T
    num_bits = _ops.convert_to_tensor(num_bits, _dtypes.int32)
    _inputs_flat = [input, input_min, input_max, num_bits]
    _attrs = ('signed_input', signed_input, 'range_given', range_given, 'T', _attr_T, 'narrow_range', narrow_range, 'axis', axis)
    _result = _execute.execute(b'QuantizeAndDequantizeV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizeAndDequantizeV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result