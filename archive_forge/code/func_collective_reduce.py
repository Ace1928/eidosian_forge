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
def collective_reduce(input: _atypes.TensorFuzzingAnnotation[TV_CollectiveReduce_T], group_size: int, group_key: int, instance_key: int, merge_op: str, final_op: str, subdiv_offsets, wait_for=[], communication_hint: str='auto', timeout_seconds: float=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveReduce_T]:
    """Mutually reduces multiple tensors of identical type and shape.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: An `int`.
    group_key: An `int`.
    instance_key: An `int`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    subdiv_offsets: A list of `ints`.
    wait_for: An optional list of `ints`. Defaults to `[]`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CollectiveReduce', name, input, 'group_size', group_size, 'group_key', group_key, 'instance_key', instance_key, 'merge_op', merge_op, 'final_op', final_op, 'subdiv_offsets', subdiv_offsets, 'wait_for', wait_for, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return collective_reduce_eager_fallback(input, group_size=group_size, group_key=group_key, instance_key=instance_key, merge_op=merge_op, final_op=final_op, subdiv_offsets=subdiv_offsets, wait_for=wait_for, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CollectiveReduce', input=input, group_size=group_size, group_key=group_key, instance_key=instance_key, merge_op=merge_op, final_op=final_op, subdiv_offsets=subdiv_offsets, wait_for=wait_for, communication_hint=communication_hint, timeout_seconds=timeout_seconds, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'group_size', _op._get_attr_int('group_size'), 'group_key', _op._get_attr_int('group_key'), 'instance_key', _op._get_attr_int('instance_key'), 'merge_op', _op.get_attr('merge_op'), 'final_op', _op.get_attr('final_op'), 'subdiv_offsets', _op.get_attr('subdiv_offsets'), 'wait_for', _op.get_attr('wait_for'), 'communication_hint', _op.get_attr('communication_hint'), 'timeout_seconds', _op.get_attr('timeout_seconds'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CollectiveReduce', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result