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
def add_many_sparse_to_tensors_map(sparse_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], sparse_values: _atypes.TensorFuzzingAnnotation[TV_AddManySparseToTensorsMap_T], sparse_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

  A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
  `sparse_values`, and `sparse_shape`, where

  ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```

  An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
  having a first `sparse_indices` column taking values between `[0, N)`, where
  the minibatch size `N == sparse_shape[0]`.

  The input `SparseTensor` must have rank `R` greater than 1, and the first
  dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The stored
  `SparseTensor` objects pointed to by each row of the output `sparse_handles`
  will have rank `R-1`.

  The `SparseTensor` values can then be read out as part of a minibatch by passing
  the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
  the correct `SparseTensorsMap` is accessed, ensure that the same
  `container` and `shared_name` are passed to that Op.  If no `shared_name`
  is provided here, instead use the *name* of the Operation created by calling
  `AddManySparseToTensorsMap` as the `shared_name` passed to
  `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
      `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
      The minibatch size `N == sparse_shape[0]`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` created by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` created by this op.
      If blank, the new Operation's unique name is used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AddManySparseToTensorsMap', name, sparse_indices, sparse_values, sparse_shape, 'container', container, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return add_many_sparse_to_tensors_map_eager_fallback(sparse_indices, sparse_values, sparse_shape, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AddManySparseToTensorsMap', sparse_indices=sparse_indices, sparse_values=sparse_values, sparse_shape=sparse_shape, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AddManySparseToTensorsMap', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result