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
def quantized_mat_mul_with_bias_eager_fallback(a: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBias_T1], b: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBias_T2], bias: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMulWithBias_Tbias], min_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], Toutput: TV_QuantizedMatMulWithBias_Toutput, transpose_a: bool, transpose_b: bool, input_quant_mode: str, name, ctx):
    if Toutput is None:
        Toutput = _dtypes.qint32
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
    _attr_T1, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_T2, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_Tbias, (bias,) = _execute.args_to_matching_eager([bias], ctx, [_dtypes.float32, _dtypes.qint32])
    min_a = _ops.convert_to_tensor(min_a, _dtypes.float32)
    max_a = _ops.convert_to_tensor(max_a, _dtypes.float32)
    min_b = _ops.convert_to_tensor(min_b, _dtypes.float32)
    max_b = _ops.convert_to_tensor(max_b, _dtypes.float32)
    _inputs_flat = [a, b, bias, min_a, max_a, min_b, max_b]
    _attrs = ('T1', _attr_T1, 'T2', _attr_T2, 'Tbias', _attr_Tbias, 'Toutput', Toutput, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'input_quant_mode', input_quant_mode)
    _result = _execute.execute(b'QuantizedMatMulWithBias', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedMatMulWithBias', _inputs_flat, _attrs, _result)
    _result = _QuantizedMatMulWithBiasOutput._make(_result)
    return _result