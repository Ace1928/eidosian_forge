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
def collective_reduce_scatter_v2(input: _atypes.TensorFuzzingAnnotation[TV_CollectiveReduceScatterV2_T], group_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], group_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], instance_key: _atypes.TensorFuzzingAnnotation[_atypes.Int32], ordering_token: List[_atypes.TensorFuzzingAnnotation[_atypes.Resource]], merge_op: str, final_op: str, communication_hint: str='auto', timeout_seconds: float=0, max_subdivs_per_device: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_CollectiveReduceScatterV2_T]:
    """Mutually reduces multiple tensors of identical type and shape and scatters the result.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`.
    group_size: A `Tensor` of type `int32`.
    group_key: A `Tensor` of type `int32`.
    instance_key: A `Tensor` of type `int32`.
    ordering_token: A list of `Tensor` objects with type `resource`.
    merge_op: A `string` from: `"Min", "Max", "Mul", "Add"`.
    final_op: A `string` from: `"Id", "Div"`.
    communication_hint: An optional `string`. Defaults to `"auto"`.
    timeout_seconds: An optional `float`. Defaults to `0`.
    max_subdivs_per_device: An optional `int`. Defaults to `-1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CollectiveReduceScatterV2', name, input, group_size, group_key, instance_key, ordering_token, 'merge_op', merge_op, 'final_op', final_op, 'communication_hint', communication_hint, 'timeout_seconds', timeout_seconds, 'max_subdivs_per_device', max_subdivs_per_device)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return collective_reduce_scatter_v2_eager_fallback(input, group_size, group_key, instance_key, ordering_token, merge_op=merge_op, final_op=final_op, communication_hint=communication_hint, timeout_seconds=timeout_seconds, max_subdivs_per_device=max_subdivs_per_device, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(ordering_token, (list, tuple)):
        raise TypeError("Expected list for 'ordering_token' argument to 'collective_reduce_scatter_v2' Op, not %r." % ordering_token)
    _attr_Nordering_token = len(ordering_token)
    merge_op = _execute.make_str(merge_op, 'merge_op')
    final_op = _execute.make_str(final_op, 'final_op')
    if communication_hint is None:
        communication_hint = 'auto'
    communication_hint = _execute.make_str(communication_hint, 'communication_hint')
    if timeout_seconds is None:
        timeout_seconds = 0
    timeout_seconds = _execute.make_float(timeout_seconds, 'timeout_seconds')
    if max_subdivs_per_device is None:
        max_subdivs_per_device = -1
    max_subdivs_per_device = _execute.make_int(max_subdivs_per_device, 'max_subdivs_per_device')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CollectiveReduceScatterV2', input=input, group_size=group_size, group_key=group_key, instance_key=instance_key, ordering_token=ordering_token, merge_op=merge_op, final_op=final_op, communication_hint=communication_hint, timeout_seconds=timeout_seconds, max_subdivs_per_device=max_subdivs_per_device, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'merge_op', _op.get_attr('merge_op'), 'final_op', _op.get_attr('final_op'), 'communication_hint', _op.get_attr('communication_hint'), 'timeout_seconds', _op.get_attr('timeout_seconds'), 'Nordering_token', _op._get_attr_int('Nordering_token'), 'max_subdivs_per_device', _op._get_attr_int('max_subdivs_per_device'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CollectiveReduceScatterV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result