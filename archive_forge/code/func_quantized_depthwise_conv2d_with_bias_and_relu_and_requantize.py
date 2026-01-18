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
def quantized_depthwise_conv2d_with_bias_and_relu_and_requantize(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tinput], filter: _atypes.TensorFuzzingAnnotation[TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tfilter], bias: _atypes.TensorFuzzingAnnotation[TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_Tbias], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], strides, padding: str, out_type: TV_QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize_out_type=_dtypes.quint8, dilations=[1, 1, 1, 1], padding_list=[], name=None):
    """Computes quantized depthwise Conv2D with Bias, Relu and Requantize.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The minimum float value of the output tensor.
    max_freezed_output: A `Tensor` of type `float32`.
      The maximum float value of the output tensor.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize', name, input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, 'out_type', out_type, 'strides', strides, 'padding', padding, 'dilations', dilations, 'padding_list', padding_list)
            _result = _QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback(input, filter, bias, min_input, max_input, min_filter, max_filter, min_freezed_output, max_freezed_output, out_type=out_type, strides=strides, padding=padding, dilations=dilations, padding_list=padding_list, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if out_type is None:
        out_type = _dtypes.quint8
    out_type = _execute.make_type(out_type, 'out_type')
    if dilations is None:
        dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    if padding_list is None:
        padding_list = []
    if not isinstance(padding_list, (list, tuple)):
        raise TypeError("Expected list for 'padding_list' argument to 'quantized_depthwise_conv2d_with_bias_and_relu_and_requantize' Op, not %r." % padding_list)
    padding_list = [_execute.make_int(_i, 'padding_list') for _i in padding_list]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize', input=input, filter=filter, bias=bias, min_input=min_input, max_input=max_input, min_filter=min_filter, max_filter=max_filter, min_freezed_output=min_freezed_output, max_freezed_output=max_freezed_output, strides=strides, padding=padding, out_type=out_type, dilations=dilations, padding_list=padding_list, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tinput', _op._get_attr_type('Tinput'), 'Tfilter', _op._get_attr_type('Tfilter'), 'Tbias', _op._get_attr_type('Tbias'), 'out_type', _op._get_attr_type('out_type'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'dilations', _op.get_attr('dilations'), 'padding_list', _op.get_attr('padding_list'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize', _inputs_flat, _attrs, _result)
    _result = _QuantizedDepthwiseConv2DWithBiasAndReluAndRequantizeOutput._make(_result)
    return _result