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
def take_many_sparse_from_tensors_map(sparse_handles: _atypes.TensorFuzzingAnnotation[_atypes.Int64], dtype: TV_TakeManySparseFromTensorsMap_dtype, container: str='', shared_name: str='', name=None):
    """Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

  The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
  `N` is the minibatch size and the rows correspond to the output handles of
  `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
  original `SparseTensor` objects that went into the given input ops must all
  match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension on the left).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the handles represent an input, which is a `[2, 3]` matrix
  representing two original `SparseTensor` objects:

  ```
      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]
  ```

  and

  ```
      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]
  ```

  then the final `SparseTensor` will be:

  ```
      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]
  ```

  Args:
    sparse_handles: A `Tensor` of type `int64`.
      1-D, The `N` serialized `SparseTensor` objects.
      Shape: `[N]`.
    dtype: A `tf.DType`.
      The `dtype` of the `SparseTensor` objects stored in the
      `SparseTensorsMap`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` read by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` read by this op.
      It should not be blank; rather the `shared_name` or unique Operation name
      of the Op that created the original `SparseTensorsMap` should be used.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TakeManySparseFromTensorsMap', name, sparse_handles, 'dtype', dtype, 'container', container, 'shared_name', shared_name)
            _result = _TakeManySparseFromTensorsMapOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return take_many_sparse_from_tensors_map_eager_fallback(sparse_handles, dtype=dtype, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TakeManySparseFromTensorsMap', sparse_handles=sparse_handles, dtype=dtype, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TakeManySparseFromTensorsMap', _inputs_flat, _attrs, _result)
    _result = _TakeManySparseFromTensorsMapOutput._make(_result)
    return _result