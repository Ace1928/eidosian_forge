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
def collective_initialize_communicator_eager_fallback(group_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], rank: _atypes.TensorFuzzingAnnotation[_atypes.Int32], group_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], communication_hint: str, timeout_seconds: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    group_key = _ops.convert_to_tensor(group_key, _dtypes.int32)
    rank = _ops.convert_to_tensor(rank, _dtypes.int32)
    group_size = _ops.convert_to_tensor(group_size, _dtypes.int32)
    _inputs_flat = [group_key, rank, group_size]
    _attrs = ('communication_hint', communication_hint, 'timeout_seconds', timeout_seconds)
    _result = _execute.execute(b'CollectiveInitializeCommunicator', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CollectiveInitializeCommunicator', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result