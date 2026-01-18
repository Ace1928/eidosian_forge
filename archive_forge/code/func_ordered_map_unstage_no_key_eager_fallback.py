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
def ordered_map_unstage_no_key_eager_fallback(indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtypes, capacity: int, memory_limit: int, container: str, shared_name: str, name, ctx):
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'ordered_map_unstage_no_key' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    if capacity is None:
        capacity = 0
    capacity = _execute.make_int(capacity, 'capacity')
    if memory_limit is None:
        memory_limit = 0
    memory_limit = _execute.make_int(memory_limit, 'memory_limit')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    indices = _ops.convert_to_tensor(indices, _dtypes.int32)
    _inputs_flat = [indices]
    _attrs = ('capacity', capacity, 'memory_limit', memory_limit, 'dtypes', dtypes, 'container', container, 'shared_name', shared_name)
    _result = _execute.execute(b'OrderedMapUnstageNoKey', len(dtypes) + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('OrderedMapUnstageNoKey', _inputs_flat, _attrs, _result)
    _result = _result[:1] + [_result[1:]]
    _result = _OrderedMapUnstageNoKeyOutput._make(_result)
    return _result