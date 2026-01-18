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
def data_service_dataset_v2(dataset_id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], processing_mode: _atypes.TensorFuzzingAnnotation[_atypes.String], address: _atypes.TensorFuzzingAnnotation[_atypes.String], protocol: _atypes.TensorFuzzingAnnotation[_atypes.String], job_name: _atypes.TensorFuzzingAnnotation[_atypes.String], consumer_index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_consumers: _atypes.TensorFuzzingAnnotation[_atypes.Int64], max_outstanding_requests: _atypes.TensorFuzzingAnnotation[_atypes.Int64], iteration_counter: _atypes.TensorFuzzingAnnotation[_atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms: int=-1, data_transfer_protocol: str='', target_workers: str='AUTO', cross_trainer_cache_options: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that reads data from the tf.data service.

  Args:
    dataset_id: A `Tensor` of type `int64`.
    processing_mode: A `Tensor` of type `string`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    job_name: A `Tensor` of type `string`.
    consumer_index: A `Tensor` of type `int64`.
    num_consumers: A `Tensor` of type `int64`.
    max_outstanding_requests: A `Tensor` of type `int64`.
    iteration_counter: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    task_refresh_interval_hint_ms: An optional `int`. Defaults to `-1`.
    data_transfer_protocol: An optional `string`. Defaults to `""`.
    target_workers: An optional `string`. Defaults to `"AUTO"`.
    cross_trainer_cache_options: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DataServiceDatasetV2', name, dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter, 'task_refresh_interval_hint_ms', task_refresh_interval_hint_ms, 'output_types', output_types, 'output_shapes', output_shapes, 'data_transfer_protocol', data_transfer_protocol, 'target_workers', target_workers, 'cross_trainer_cache_options', cross_trainer_cache_options)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return data_service_dataset_v2_eager_fallback(dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, output_types=output_types, output_shapes=output_shapes, data_transfer_protocol=data_transfer_protocol, target_workers=target_workers, cross_trainer_cache_options=cross_trainer_cache_options, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'data_service_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'data_service_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if task_refresh_interval_hint_ms is None:
        task_refresh_interval_hint_ms = -1
    task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, 'task_refresh_interval_hint_ms')
    if data_transfer_protocol is None:
        data_transfer_protocol = ''
    data_transfer_protocol = _execute.make_str(data_transfer_protocol, 'data_transfer_protocol')
    if target_workers is None:
        target_workers = 'AUTO'
    target_workers = _execute.make_str(target_workers, 'target_workers')
    if cross_trainer_cache_options is None:
        cross_trainer_cache_options = ''
    cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, 'cross_trainer_cache_options')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DataServiceDatasetV2', dataset_id=dataset_id, processing_mode=processing_mode, address=address, protocol=protocol, job_name=job_name, consumer_index=consumer_index, num_consumers=num_consumers, max_outstanding_requests=max_outstanding_requests, iteration_counter=iteration_counter, output_types=output_types, output_shapes=output_shapes, task_refresh_interval_hint_ms=task_refresh_interval_hint_ms, data_transfer_protocol=data_transfer_protocol, target_workers=target_workers, cross_trainer_cache_options=cross_trainer_cache_options, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('task_refresh_interval_hint_ms', _op._get_attr_int('task_refresh_interval_hint_ms'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'data_transfer_protocol', _op.get_attr('data_transfer_protocol'), 'target_workers', _op.get_attr('target_workers'), 'cross_trainer_cache_options', _op.get_attr('cross_trainer_cache_options'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DataServiceDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result