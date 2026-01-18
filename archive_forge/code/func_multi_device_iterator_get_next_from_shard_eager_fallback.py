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
def multi_device_iterator_get_next_from_shard_eager_fallback(multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], shard_num: _atypes.TensorFuzzingAnnotation[_atypes.Int32], incarnation_id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, name, ctx):
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'multi_device_iterator_get_next_from_shard' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'multi_device_iterator_get_next_from_shard' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
    shard_num = _ops.convert_to_tensor(shard_num, _dtypes.int32)
    incarnation_id = _ops.convert_to_tensor(incarnation_id, _dtypes.int64)
    _inputs_flat = [multi_device_iterator, shard_num, incarnation_id]
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'MultiDeviceIteratorGetNextFromShard', len(output_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MultiDeviceIteratorGetNextFromShard', _inputs_flat, _attrs, _result)
    return _result