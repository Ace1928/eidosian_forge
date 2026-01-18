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
def ragged_cross(ragged_values, ragged_row_splits, sparse_indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], sparse_values, sparse_shape: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], dense_inputs, input_order: str, hashed_output: bool, num_buckets: int, hash_key: int, out_values_type: TV_RaggedCross_out_values_type, out_row_splits_type: TV_RaggedCross_out_row_splits_type, name=None):
    """Generates a feature cross from a list of tensors, and returns it as a
RaggedTensor.  See `tf.ragged.cross` for more details.

  Args:
    ragged_values: A list of `Tensor` objects with types from: `int64`, `string`.
      The values tensor for each RaggedTensor input.
    ragged_row_splits: A list of `Tensor` objects with types from: `int32`, `int64`.
      The row_splits tensor for each RaggedTensor input.
    sparse_indices: A list of `Tensor` objects with type `int64`.
      The indices tensor for each SparseTensor input.
    sparse_values: A list of `Tensor` objects with types from: `int64`, `string`.
      The values tensor for each SparseTensor input.
    sparse_shape: A list with the same length as `sparse_indices` of `Tensor` objects with type `int64`.
      The dense_shape tensor for each SparseTensor input.
    dense_inputs: A list of `Tensor` objects with types from: `int64`, `string`.
      The tf.Tensor inputs.
    input_order: A `string`.
      String specifying the tensor type for each input.  The `i`th character in
      this string specifies the type of the `i`th input, and is one of: 'R' (ragged),
      'D' (dense), or 'S' (sparse).  This attr is used to ensure that the crossed
      values are combined in the order of the inputs from the call to tf.ragged.cross.
    hashed_output: A `bool`.
    num_buckets: An `int` that is `>= 0`.
    hash_key: An `int`.
    out_values_type: A `tf.DType` from: `tf.int64, tf.string`.
    out_row_splits_type: A `tf.DType` from: `tf.int32, tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_values, output_row_splits).

    output_values: A `Tensor` of type `out_values_type`.
    output_row_splits: A `Tensor` of type `out_row_splits_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedCross', name, ragged_values, ragged_row_splits, sparse_indices, sparse_values, sparse_shape, dense_inputs, 'input_order', input_order, 'hashed_output', hashed_output, 'num_buckets', num_buckets, 'hash_key', hash_key, 'out_values_type', out_values_type, 'out_row_splits_type', out_row_splits_type)
            _result = _RaggedCrossOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_cross_eager_fallback(ragged_values, ragged_row_splits, sparse_indices, sparse_values, sparse_shape, dense_inputs, input_order=input_order, hashed_output=hashed_output, num_buckets=num_buckets, hash_key=hash_key, out_values_type=out_values_type, out_row_splits_type=out_row_splits_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(sparse_indices, (list, tuple)):
        raise TypeError("Expected list for 'sparse_indices' argument to 'ragged_cross' Op, not %r." % sparse_indices)
    _attr_Nsparse = len(sparse_indices)
    if not isinstance(sparse_shape, (list, tuple)):
        raise TypeError("Expected list for 'sparse_shape' argument to 'ragged_cross' Op, not %r." % sparse_shape)
    if len(sparse_shape) != _attr_Nsparse:
        raise ValueError("List argument 'sparse_shape' to 'ragged_cross' Op with length %d must match length %d of argument 'sparse_indices'." % (len(sparse_shape), _attr_Nsparse))
    input_order = _execute.make_str(input_order, 'input_order')
    hashed_output = _execute.make_bool(hashed_output, 'hashed_output')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    hash_key = _execute.make_int(hash_key, 'hash_key')
    out_values_type = _execute.make_type(out_values_type, 'out_values_type')
    out_row_splits_type = _execute.make_type(out_row_splits_type, 'out_row_splits_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedCross', ragged_values=ragged_values, ragged_row_splits=ragged_row_splits, sparse_indices=sparse_indices, sparse_values=sparse_values, sparse_shape=sparse_shape, dense_inputs=dense_inputs, input_order=input_order, hashed_output=hashed_output, num_buckets=num_buckets, hash_key=hash_key, out_values_type=out_values_type, out_row_splits_type=out_row_splits_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Nsparse', _op._get_attr_int('Nsparse'), 'input_order', _op.get_attr('input_order'), 'hashed_output', _op._get_attr_bool('hashed_output'), 'num_buckets', _op._get_attr_int('num_buckets'), 'hash_key', _op._get_attr_int('hash_key'), 'ragged_values_types', _op.get_attr('ragged_values_types'), 'ragged_splits_types', _op.get_attr('ragged_splits_types'), 'sparse_values_types', _op.get_attr('sparse_values_types'), 'dense_types', _op.get_attr('dense_types'), 'out_values_type', _op._get_attr_type('out_values_type'), 'out_row_splits_type', _op._get_attr_type('out_row_splits_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedCross', _inputs_flat, _attrs, _result)
    _result = _RaggedCrossOutput._make(_result)
    return _result