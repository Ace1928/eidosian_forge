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
def cross_replica_sum_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CrossReplicaSum_T], group_assignment: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CrossReplicaSum_T]:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint32])
    group_assignment = _ops.convert_to_tensor(group_assignment, _dtypes.int32)
    _inputs_flat = [input, group_assignment]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'CrossReplicaSum', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CrossReplicaSum', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result