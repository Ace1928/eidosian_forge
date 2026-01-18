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
def ragged_tensor_to_variant_eager_fallback(rt_nested_splits: List[_atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariant_Tsplits]], rt_dense_values: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariant_Tvalues], batched_input: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(rt_nested_splits, (list, tuple)):
        raise TypeError("Expected list for 'rt_nested_splits' argument to 'ragged_tensor_to_variant' Op, not %r." % rt_nested_splits)
    _attr_RAGGED_RANK = len(rt_nested_splits)
    batched_input = _execute.make_bool(batched_input, 'batched_input')
    _attr_Tvalues, (rt_dense_values,) = _execute.args_to_matching_eager([rt_dense_values], ctx, [])
    _attr_Tsplits, rt_nested_splits = _execute.args_to_matching_eager(list(rt_nested_splits), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    _inputs_flat = list(rt_nested_splits) + [rt_dense_values]
    _attrs = ('RAGGED_RANK', _attr_RAGGED_RANK, 'Tvalues', _attr_Tvalues, 'Tsplits', _attr_Tsplits, 'batched_input', batched_input)
    _result = _execute.execute(b'RaggedTensorToVariant', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedTensorToVariant', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result