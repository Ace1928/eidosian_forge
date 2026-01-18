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
def quantized_mat_mul(a: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMul_T1], b: _atypes.TensorFuzzingAnnotation[TV_QuantizedMatMul_T2], min_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_b: _atypes.TensorFuzzingAnnotation[_atypes.Float32], Toutput: TV_QuantizedMatMul_Toutput=_dtypes.qint32, transpose_a: bool=False, transpose_b: bool=False, Tactivation: TV_QuantizedMatMul_Tactivation=_dtypes.quint8, name=None):
    """Perform a quantized matrix multiplication of  `a` by the matrix `b`.

  The inputs must be two-dimensional matrices and the inner dimension of
  `a` (after being transposed if `transpose_a` is non-zero) must match the
  outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero).

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    Tactivation: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of output produced by activation function
      following this operation.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedMatMul', name, a, b, min_a, max_a, min_b, max_b, 'Toutput', Toutput, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'Tactivation', Tactivation)
            _result = _QuantizedMatMulOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_mat_mul_eager_fallback(a, b, min_a, max_a, min_b, max_b, Toutput=Toutput, transpose_a=transpose_a, transpose_b=transpose_b, Tactivation=Tactivation, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Toutput is None:
        Toutput = _dtypes.qint32
    Toutput = _execute.make_type(Toutput, 'Toutput')
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if Tactivation is None:
        Tactivation = _dtypes.quint8
    Tactivation = _execute.make_type(Tactivation, 'Tactivation')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedMatMul', a=a, b=b, min_a=min_a, max_a=max_a, min_b=min_b, max_b=max_b, Toutput=Toutput, transpose_a=transpose_a, transpose_b=transpose_b, Tactivation=Tactivation, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T1', _op._get_attr_type('T1'), 'T2', _op._get_attr_type('T2'), 'Toutput', _op._get_attr_type('Toutput'), 'transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'Tactivation', _op._get_attr_type('Tactivation'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedMatMul', _inputs_flat, _attrs, _result)
    _result = _QuantizedMatMulOutput._make(_result)
    return _result