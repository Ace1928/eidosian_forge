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
def collective_gather_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CollectiveGatherV2_T], group_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], group_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], instance_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], ordering_token: List[_atypes.TensorFuzzingAnnotation[_atypes.Resource]], communication_hint: str, timeout_seconds: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveGatherV2_T]:
    if not isinstance(ordering_token, (list, tuple)):
        raise TypeError("Expected list for 'ordering_token' argument to 'collective_gather_v2' Op, not %r." % ordering_token)
    _attr_Nordering_token = len(ordering_token)
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64])
    group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
    group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
    instance_key = _ops.convert_to_tensor(instance_key, _dtypes.int32)
    ordering_token = _ops.convert_n_to_tensor(ordering_token, _dtypes.resource)
    _inputs_flat = [input, group_size, group_key, instance_key] + list(ordering_token)
    _attrs = ('T', _attr_T, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds, 'Nordering_token', _attr_Nordering_token)
    _result = _execute.execute(b'CollectiveGatherV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CollectiveGatherV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result