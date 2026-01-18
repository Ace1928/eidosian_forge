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
def ragged_tensor_to_sparse(rt_nested_splits: List[_atypes.TensorFuzzingAnnotation[TV_RaggedTensorToSparse_Tsplits]], rt_dense_values: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToSparse_T], name=None):
    """Converts a `RaggedTensor` into a `SparseTensor` with the same values.

  input=ragged.from_nested_row_splits(rt_dense_values, rt_nested_splits)
  output=SparseTensor(indices=sparse_indices, values=sparse_values,
                      dense_shape=sparse_dense_shape)

  Args:
    rt_nested_splits: A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
      The `row_splits` for the `RaggedTensor`.
    rt_dense_values: A `Tensor`. The `flat_values` for the `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_dense_shape).

    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor`. Has the same type as `rt_dense_values`.
    sparse_dense_shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedTensorToSparse', name, rt_nested_splits, rt_dense_values)
            _result = _RaggedTensorToSparseOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_tensor_to_sparse_eager_fallback(rt_nested_splits, rt_dense_values, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(rt_nested_splits, (list, tuple)):
        raise TypeError("Expected list for 'rt_nested_splits' argument to 'ragged_tensor_to_sparse' Op, not %r." % rt_nested_splits)
    _attr_RAGGED_RANK = len(rt_nested_splits)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedTensorToSparse', rt_nested_splits=rt_nested_splits, rt_dense_values=rt_dense_values, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('RAGGED_RANK', _op._get_attr_int('RAGGED_RANK'), 'T', _op._get_attr_type('T'), 'Tsplits', _op._get_attr_type('Tsplits'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedTensorToSparse', _inputs_flat, _attrs, _result)
    _result = _RaggedTensorToSparseOutput._make(_result)
    return _result