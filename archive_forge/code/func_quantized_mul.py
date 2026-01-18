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
def quantized_mul(x: _atypes.TensorFuzzingAnnotation[TV_QuantizedMul_T1], y: _atypes.TensorFuzzingAnnotation[TV_QuantizedMul_T2], min_x: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_x: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_y: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_y: _atypes.TensorFuzzingAnnotation[_atypes.Float32], Toutput: TV_QuantizedMul_Toutput=_dtypes.qint32, name=None):
    """Returns x * y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedMul', name, x, y, min_x, max_x, min_y, max_y, 'Toutput', Toutput)
            _result = _QuantizedMulOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_mul_eager_fallback(x, y, min_x, max_x, min_y, max_y, Toutput=Toutput, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Toutput is None:
        Toutput = _dtypes.qint32
    Toutput = _execute.make_type(Toutput, 'Toutput')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedMul', x=x, y=y, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, Toutput=Toutput, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T1', _op._get_attr_type('T1'), 'T2', _op._get_attr_type('T2'), 'Toutput', _op._get_attr_type('Toutput'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedMul', _inputs_flat, _attrs, _result)
    _result = _QuantizedMulOutput._make(_result)
    return _result