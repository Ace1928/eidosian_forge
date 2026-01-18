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
def quantized_mat_mul_with_bias_and_dequantize(a: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBiasAndDequantize_T1], b: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBiasAndDequantize_T2], bias: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBiasAndDequantize_Tbias], min_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_freezed_output: _atypes.TensorFuzzingAnnotation[_atypes.Float32], Toutput: TV_QuantizedMatMulWithBiasAndDequantize_Toutput, transpose_a: bool=False, transpose_b: bool=False, input_quant_mode: str='MIN_FIRST', name=None) -> _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBiasAndDequantize_Toutput]:
    """TODO: add doc.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_a: A `Tensor` of type `float32`.
    max_a: A `Tensor` of type `float32`.
    min_b: A `Tensor` of type `float32`.
    max_b: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: A `tf.DType` from: `tf.float32`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    input_quant_mode: An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Toutput`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedMatMulWithBiasAndDequantize', name, a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output, 'Toutput', Toutput, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'input_quant_mode', input_quant_mode)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_mat_mul_with_bias_and_dequantize_eager_fallback(a, b, bias, min_a, max_a, min_b, max_b, min_freezed_output, max_freezed_output, Toutput=Toutput, transpose_a=transpose_a, transpose_b=transpose_b, input_quant_mode=input_quant_mode, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Toutput = _execute.make_type(Toutput, 'Toutput')
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if input_quant_mode is None:
        input_quant_mode = 'MIN_FIRST'
    input_quant_mode = _execute.make_str(input_quant_mode, 'input_quant_mode')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedMatMulWithBiasAndDequantize', a=a, b=b, bias=bias, min_a=min_a, max_a=max_a, min_b=min_b, max_b=max_b, min_freezed_output=min_freezed_output, max_freezed_output=max_freezed_output, Toutput=Toutput, transpose_a=transpose_a, transpose_b=transpose_b, input_quant_mode=input_quant_mode, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T1', _op._get_attr_type('T1'), 'T2', _op._get_attr_type('T2'), 'Tbias', _op._get_attr_type('Tbias'), 'Toutput', _op._get_attr_type('Toutput'), 'transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'input_quant_mode', _op.get_attr('input_quant_mode'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedMatMulWithBiasAndDequantize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result