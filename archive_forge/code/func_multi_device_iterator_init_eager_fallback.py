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
def multi_device_iterator_init_eager_fallback(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], max_buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
    multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
    max_buffer_size = _ops.convert_to_tensor(max_buffer_size, _dtypes.int64)
    _inputs_flat = [dataset, multi_device_iterator, max_buffer_size]
    _attrs = None
    _result = _execute.execute(b'MultiDeviceIteratorInit', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MultiDeviceIteratorInit', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result