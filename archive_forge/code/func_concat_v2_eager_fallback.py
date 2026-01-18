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
def concat_v2_eager_fallback(values: List[_atypes.TensorFuzzingAnnotation[TV_ConcatV2_T]], axis: _atypes.TensorFuzzingAnnotation[TV_ConcatV2_Tidx], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ConcatV2_T]:
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'concat_v2' Op, not %r." % values)
    _attr_N = len(values)
    _attr_T, values = _execute.args_to_matching_eager(list(values), ctx, [])
    _attr_Tidx, (axis,) = _execute.args_to_matching_eager([axis], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = list(values) + [axis]
    _attrs = ('N', _attr_N, 'T', _attr_T, 'Tidx', _attr_Tidx)
    _result = _execute.execute(b'ConcatV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ConcatV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result