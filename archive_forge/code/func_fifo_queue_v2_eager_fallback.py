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
def fifo_queue_v2_eager_fallback(component_types, shapes, capacity: int, container: str, shared_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'fifo_queue_v2' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if shapes is None:
        shapes = []
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'fifo_queue_v2' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if capacity is None:
        capacity = -1
    capacity = _execute.make_int(capacity, 'capacity')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _inputs_flat = []
    _attrs = ('component_types', component_types, 'shapes', shapes, 'capacity', capacity, 'container', container, 'shared_name', shared_name)
    _result = _execute.execute(b'FIFOQueueV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FIFOQueueV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result