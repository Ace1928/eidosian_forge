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
def queue_dequeue_v2_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], component_types, timeout_ms: int, name, ctx):
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'queue_dequeue_v2' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if timeout_ms is None:
        timeout_ms = -1
    timeout_ms = _execute.make_int(timeout_ms, 'timeout_ms')
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    _inputs_flat = [handle]
    _attrs = ('component_types', component_types, 'timeout_ms', timeout_ms)
    _result = _execute.execute(b'QueueDequeueV2', len(component_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QueueDequeueV2', _inputs_flat, _attrs, _result)
    return _result