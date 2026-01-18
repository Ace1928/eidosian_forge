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
def sparse_segment_sum_grad_eager_fallback(grad: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSumGrad_T], indices: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSumGrad_Tidx], segment_ids: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSumGrad_Tsegmentids], output_dim0: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSumGrad_T]:
    _attr_T, (grad,) = _execute.args_to_matching_eager([grad], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    _attr_Tidx, (indices,) = _execute.args_to_matching_eager([indices], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tsegmentids, (segment_ids,) = _execute.args_to_matching_eager([segment_ids], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    output_dim0 = _ops.convert_to_tensor(output_dim0, _dtypes.int32)
    _inputs_flat = [grad, indices, segment_ids, output_dim0]
    _attrs = ('T', _attr_T, 'Tidx', _attr_Tidx, 'Tsegmentids', _attr_Tsegmentids)
    _result = _execute.execute(b'SparseSegmentSumGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseSegmentSumGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result