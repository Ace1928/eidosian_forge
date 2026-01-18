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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('bitwise.right_shift')
def right_shift(x: _atypes.TensorFuzzingAnnotation[TV_RightShift_T], y: _atypes.TensorFuzzingAnnotation[TV_RightShift_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_RightShift_T]:
    """Elementwise computes the bitwise right-shift of `x` and `y`.

  Performs a logical shift for unsigned integer types, and an arithmetic shift
  for signed integer types.

  If `y` is negative, or greater than or equal to than the width of `x` in bits
  the result is implementation defined.

  Example:

  ```python
  import tensorflow as tf
  from tensorflow.python.ops import bitwise_ops
  import numpy as np
  dtype_list = [tf.int8, tf.int16, tf.int32, tf.int64]

  for dtype in dtype_list:
    lhs = tf.constant([-1, -5, -3, -14], dtype=dtype)
    rhs = tf.constant([5, 0, 7, 11], dtype=dtype)

    right_shift_result = bitwise_ops.right_shift(lhs, rhs)

    print(right_shift_result)

  # This will print:
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int8)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int16)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int32)
  # tf.Tensor([-1 -5 -1 -1], shape=(4,), dtype=int64)

  lhs = np.array([-2, 64, 101, 32], dtype=np.int8)
  rhs = np.array([-1, -5, -3, -14], dtype=np.int8)
  bitwise_ops.right_shift(lhs, rhs)
  # <tf.Tensor: shape=(4,), dtype=int8, numpy=array([ -2,  64, 101,  32], dtype=int8)>
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RightShift', name, x, y)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_right_shift((x, y, name), None)
            if _result is not NotImplemented:
                return _result
            return right_shift_eager_fallback(x, y, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(right_shift, (), dict(x=x, y=y, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_right_shift((x, y, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RightShift', x=x, y=y, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(right_shift, (), dict(x=x, y=y, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RightShift', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result