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
def sparse_segment_sqrt_n_grad_v2(grad: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNGradV2_T], indices: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNGradV2_Tidx], segment_ids: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNGradV2_Tsegmentids], dense_output_dim0: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None):
    """Computes gradients for SparseSegmentSqrtN.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is the number of unique indexes in "indices". Also returns vector
  "sorted_unique_indices" containing the corresponding indexes from "indices".

  Args:
    grad: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    dense_output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, sorted_unique_indices).

    output: A `Tensor`. Has the same type as `grad`.
    sorted_unique_indices: A `Tensor`. Has the same type as `indices`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseSegmentSqrtNGradV2', name, grad, indices, segment_ids, dense_output_dim0)
            _result = _SparseSegmentSqrtNGradV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_segment_sqrt_n_grad_v2_eager_fallback(grad, indices, segment_ids, dense_output_dim0, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseSegmentSqrtNGradV2', grad=grad, indices=indices, segment_ids=segment_ids, dense_output_dim0=dense_output_dim0, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tidx', _op._get_attr_type('Tidx'), 'Tsegmentids', _op._get_attr_type('Tsegmentids'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseSegmentSqrtNGradV2', _inputs_flat, _attrs, _result)
    _result = _SparseSegmentSqrtNGradV2Output._make(_result)
    return _result