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
def cumulative_logsumexp_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_CumulativeLogsumexp_T], axis: _atypes.TensorFuzzingAnnotation[TV_CumulativeLogsumexp_Tidx], exclusive: bool, reverse: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CumulativeLogsumexp_T]:
    if exclusive is None:
        exclusive = False
    exclusive = _execute.make_bool(exclusive, 'exclusive')
    if reverse is None:
        reverse = False
    reverse = _execute.make_bool(reverse, 'reverse')
    _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [x, axis]
    _attrs = ('exclusive', exclusive, 'reverse', reverse, 'T', _attr_T, 'Tidx', _attr_Tidx)
    _result = _execute.execute(b'CumulativeLogsumexp', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CumulativeLogsumexp', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result