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
def tensor_scatter_update(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorScatterUpdate_T], indices: _atypes.TensorFuzzingAnnotation[TV_TensorScatterUpdate_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_TensorScatterUpdate_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorScatterUpdate_T]:
    """Scatter `updates` into an existing tensor according to `indices`.

  This operation creates a new tensor by applying sparse `updates` to the passed
  in `tensor`.
  This operation is very similar to `tf.scatter_nd`, except that the updates are
  scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  If `indices` contains duplicates, then we pick the last update for the index.

  If an out of bound index is found on CPU, an error is returned.

  **WARNING**: There are some GPU specific semantics for this operation.
  - If an out of bound index is found, the index is ignored.
  - The order in which updates are applied is nondeterministic, so the output
  will be nondeterministic if `indices` contains duplicates.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.

  * `indices` must have at least 2 axes: `(num_updates, index_depth)`.
  * The last axis of `indices` is how deep to index into `tensor` so  this index
    depth must be less than the rank of `tensor`: `indices.shape[-1] <= tensor.ndim`

  if `indices.shape[-1] = tensor.rank` this Op indexes and updates scalar elements.
  if `indices.shape[-1] < tensor.rank` it indexes and updates slices of the input
  `tensor`.

  Each `update` has a rank of `tensor.rank - indices.shape[-1]`.
  The overall shape of `updates` is:

  ```
  indices.shape[:-1] + tensor.shape[indices.shape[-1]:]
  ```

  For usage examples see the python [tf.tensor_scatter_nd_update](
  https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update) function

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int16`, `int32`, `int64`, `uint16`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorScatterUpdate', name, tensor, indices, updates)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_scatter_update_eager_fallback(tensor, indices, updates, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorScatterUpdate', tensor=tensor, indices=indices, updates=updates, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorScatterUpdate', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result