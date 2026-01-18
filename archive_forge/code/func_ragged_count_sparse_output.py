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
def ragged_count_sparse_output(splits: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_RaggedCountSparseOutput_T], weights: _atypes.TensorFuzzingAnnotation[TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength: int=-1, maxlength: int=-1, name=None):
    """Performs sparse-output bin counting for a ragged tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    splits: A `Tensor` of type `int64`.
      Tensor containing the row splits of the ragged tensor to count.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing values of the sparse tensor to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values.
      May also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedCountSparseOutput', name, splits, values, weights, 'minlength', minlength, 'maxlength', maxlength, 'binary_output', binary_output)
            _result = _RaggedCountSparseOutputOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_count_sparse_output_eager_fallback(splits, values, weights, minlength=minlength, maxlength=maxlength, binary_output=binary_output, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    binary_output = _execute.make_bool(binary_output, 'binary_output')
    if minlength is None:
        minlength = -1
    minlength = _execute.make_int(minlength, 'minlength')
    if maxlength is None:
        maxlength = -1
    maxlength = _execute.make_int(maxlength, 'maxlength')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedCountSparseOutput', splits=splits, values=values, weights=weights, binary_output=binary_output, minlength=minlength, maxlength=maxlength, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'minlength', _op._get_attr_int('minlength'), 'maxlength', _op._get_attr_int('maxlength'), 'binary_output', _op._get_attr_bool('binary_output'), 'output_type', _op._get_attr_type('output_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedCountSparseOutput', _inputs_flat, _attrs, _result)
    _result = _RaggedCountSparseOutputOutput._make(_result)
    return _result