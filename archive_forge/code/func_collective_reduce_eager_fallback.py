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
def collective_reduce_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CollectiveReduce_T], group_size: int, group_key: int, instance_key: int, merge_op: str, final_op: str, subdiv_offsets, wait_for, communication_hint: str, timeout_seconds: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveReduce_T]:
    group_size = _execute.make_int(group_size, 'group_size')
    group_key = _execute.make_int(group_key, 'group_key')
    instance_key = _execute.make_int(instance_key, 'instance_key')
    merge_op = _execute.make_str(merge_op, 'merge_op')
    final_op = _execute.make_str(final_op, 'final_op')
    if not isinstance(subdiv_offsets, (list, tuple)):
        raise TypeError("Expected list for 'subdiv_offsets' argument to 'collective_reduce' Op, not %r." % subdiv_offsets)
    subdiv_offsets = [_execute.make_int(_i, 'subdiv_offsets') for _i in subdiv_offsets]
    if wait_for is None:
        wait_for = []
    if not isinstance(wait_for, (list, tuple)):
        raise TypeError("Expected list for 'wait_for' argument to 'collective_reduce' Op, not %r." % wait_for)
    wait_for = [_execute.make_int(_i, 'wait_for') for _i in wait_for]
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.bfloat16, _dtypes.float32, _dtypes.half, _dtypes.float64, _dtypes.int32, _dtypes.int64])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'group_size', group_size, 'group_key', group_key, 'instance_key', instance_key, 'merge_op', merge_op, 'final_op', final_op, 'subdiv_offsets', subdiv_offsets, 'wait_for', wait_for, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds)
    _result = _execute.execute(b'CollectiveReduce', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CollectiveReduce', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result