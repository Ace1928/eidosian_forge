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
def quantized_conv2d_per_channel(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DPerChannel_Tinput], filter: _atypes.TensorFuzzingAnnotation[TV_QuantizedConv2DPerChannel_Tfilter], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_filter: _atypes.TensorFuzzingAnnotation[_atypes.Float32], strides, padding: str, out_type: TV_QuantizedConv2DPerChannel_out_type=_dtypes.qint32, dilations=[1, 1, 1, 1], name=None):
    """Computes QuantizedConv2D per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    max_input: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    min_filter: A `Tensor` of type `float32`.
      The minimum value of the filter tensor.
    max_filter: A `Tensor` of type `float32`.
      The maximum value of the filter tensor.
    strides: A list of `ints`. list of stride values.
    padding: A `string` from: `"SAME", "VALID"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The quantized type of output tensor that needs to be converted.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      list of dilation values.
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedConv2DPerChannel', name, input, filter, min_input, max_input, min_filter, max_filter, 'out_type', out_type, 'strides', strides, 'padding', padding, 'dilations', dilations)
            _result = _QuantizedConv2DPerChannelOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_conv2d_per_channel_eager_fallback(input, filter, min_input, max_input, min_filter, max_filter, out_type=out_type, strides=strides, padding=padding, dilations=dilations, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'quantized_conv2d_per_channel' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if out_type is None:
        out_type = _dtypes.qint32
    out_type = _execute.make_type(out_type, 'out_type')
    if dilations is None:
        dilations = [1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'quantized_conv2d_per_channel' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedConv2DPerChannel', input=input, filter=filter, min_input=min_input, max_input=max_input, min_filter=min_filter, max_filter=max_filter, strides=strides, padding=padding, out_type=out_type, dilations=dilations, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tinput', _op._get_attr_type('Tinput'), 'Tfilter', _op._get_attr_type('Tfilter'), 'out_type', _op._get_attr_type('out_type'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'dilations', _op.get_attr('dilations'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedConv2DPerChannel', _inputs_flat, _attrs, _result)
    _result = _QuantizedConv2DPerChannelOutput._make(_result)
    return _result