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
def quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tinput], filter: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tfilter], bias: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tbias], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], summand: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_Tsummand], min_summand: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_summand: _atypes.TensorFuzzingAnnotation[_atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DWithBiasSumAndReluAndRequantize_out_type, dilations, padding_list, name, ctx):
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if out_type is None:
        out_type = _dtypes.quint8
    out_type = _execute.make_type(out_type, 'out_type')
    if dilations is None:
        dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    if padding_list is None:
        padding_list = []
    if not isinstance(padding_list, (list, tuple)):
        raise TypeError("Expected list for 'padding_list' argument to 'quantized_conv2d_with_bias_sum_and_relu_and_requantize' Op, not %r." % padding_list)
    padding_list = [_execute.make_int(_i, 'padding_list') for _i in padding_list]
    _attr_Tinput, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_Tfilter, (filter,) = _execute.args_to_matching_eager([filter], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32])
    _attr_Tsummand, (summand,) = _execute.args_to_matching_eager([summand], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
    max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
    min_filter = _ops.convert_to_tensor(min_filter, _dtypes.float32)
    max_filter = _ops.convert_to_tensor(max_filter, _dtypes.float32)
    min_freezed_output = _ops.convert_to_tensor(min_freezed_output, _dtypes.float32)
    max_freezed_output = _ops.convert_to_tensor(max_freezed_output, _dtypes.float32)
    min_summand = _ops.convert_to_tensor(min_summand, _dtypes.float32)
    max_summand = _ops.convert_to_tensor(max_summand, _dtypes.float32)
    _inputs_flat = [input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, summand, min_summand, max_summand]
    _attrs = ('Tinput', _attr_Tinput, 'Tfilter', _attr_Tfilter, 'Tbias', _attr_Tbias, 'Tsummand', _attr_Tsummand, 'out_type', out_type, 'strides', strides, 'padding', padding, 'dilations', dilations, 'padding_list', padding_list)
    _result = _execute.execute(b'QuantizedConv2DWithBiasSumAndReluAndRequantize', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedConv2DWithBiasSumAndReluAndRequantize', _inputs_flat, _attrs, _result)
    _result = _QuantizedConv2DWithBiasSumAndReluAndRequantizeOutput._make(_result)
    return _result