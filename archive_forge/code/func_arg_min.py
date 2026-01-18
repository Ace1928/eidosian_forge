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
def arg_min(input: _atypes.TensorFuzzingAnnotation[TV_ArgMin_T], dimension: _atypes.TensorFuzzingAnnotation[TV_ArgMin_Tidx], output_type: TV_ArgMin_output_type=_dtypes.int64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ArgMin_output_type]:
    """Returns the index with the smallest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmin(input = a)
    c = tf.keras.backend.eval(b)
    # c = 0
    # here a[0] = 1 which is the smallest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`, `bool`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ArgMin', name, input, dimension, 'output_type', output_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return arg_min_eager_fallback(input, dimension, output_type=output_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_type is None:
        output_type = _dtypes.int64
    output_type = _execute.make_type(output_type, 'output_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ArgMin', input=input, dimension=dimension, output_type=output_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tidx', _op._get_attr_type('Tidx'), 'output_type', _op._get_attr_type('output_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ArgMin', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result