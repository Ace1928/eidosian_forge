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
@tf_export('dynamic_stitch')
def dynamic_stitch(indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], data: List[_atypes.TensorFuzzingAnnotation[TV_DynamicStitch_T]], name=None) -> _atypes.TensorFuzzingAnnotation[TV_DynamicStitch_T]:
    """Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

  ```python
      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
  ```

  For example, if each `indices[m]` is scalar or vector, we have

  ```python
      # Scalar indices:
      merged[indices[m], ...] = data[m][...]

      # Vector indices:
      merged[indices[m][i], ...] = data[m][i, ...]
  ```

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result. If you do not need this guarantee, ParallelDynamicStitch might
  perform better on some devices.

  For example:

  ```python
      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]
  ```

  This method can be used to merge partitions created by `dynamic_partition`
  as illustrated on the following example:

  ```python
      # Apply function (increments x_i) on elements for which a certain condition
      # apply (x_i != -1 in this example).
      x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
      condition_mask=tf.not_equal(x,tf.constant(-1.))
      partitioned_data = tf.dynamic_partition(
          x, tf.cast(condition_mask, tf.int32) , 2)
      partitioned_data[1] = partitioned_data[1] + 1.0
      condition_indices = tf.dynamic_partition(
          tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
      x = tf.dynamic_stitch(condition_indices, partitioned_data)
      # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
      # unchanged.
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 1 `Tensor` objects with type `int32`.
    data: A list with the same length as `indices` of `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DynamicStitch', name, indices, data)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_dynamic_stitch((indices, data, name), None)
            if _result is not NotImplemented:
                return _result
            return dynamic_stitch_eager_fallback(indices, data, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(dynamic_stitch, (), dict(indices=indices, data=data, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_dynamic_stitch((indices, data, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(indices, (list, tuple)):
        raise TypeError("Expected list for 'indices' argument to 'dynamic_stitch' Op, not %r." % indices)
    _attr_N = len(indices)
    if not isinstance(data, (list, tuple)):
        raise TypeError("Expected list for 'data' argument to 'dynamic_stitch' Op, not %r." % data)
    if len(data) != _attr_N:
        raise ValueError("List argument 'data' to 'dynamic_stitch' Op with length %d must match length %d of argument 'indices'." % (len(data), _attr_N))
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('DynamicStitch', indices=indices, data=data, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(dynamic_stitch, (), dict(indices=indices, data=data, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DynamicStitch', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result