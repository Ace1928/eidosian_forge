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
def parallel_dynamic_stitch_eager_fallback(indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], data: List[_atypes.TensorFuzzingAnnotation[TV_ParallelDynamicStitch_T]], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ParallelDynamicStitch_T]:
    if not isinstance(indices, (list, tuple)):
        raise TypeError("Expected list for 'indices' argument to 'parallel_dynamic_stitch' Op, not %r." % indices)
    _attr_N = len(indices)
    if not isinstance(data, (list, tuple)):
        raise TypeError("Expected list for 'data' argument to 'parallel_dynamic_stitch' Op, not %r." % data)
    if len(data) != _attr_N:
        raise ValueError("List argument 'data' to 'parallel_dynamic_stitch' Op with length %d must match length %d of argument 'indices'." % (len(data), _attr_N))
    _attr_T, data = _execute.args_to_matching_eager(list(data), ctx, [])
    indices = _ops.convert_n_to_tensor(indices, _dtypes.int32)
    _inputs_flat = list(indices) + list(data)
    _attrs = ('N', _attr_N, 'T', _attr_T)
    _result = _execute.execute(b'ParallelDynamicStitch', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParallelDynamicStitch', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result