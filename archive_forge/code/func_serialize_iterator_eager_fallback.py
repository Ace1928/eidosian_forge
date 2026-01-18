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
def serialize_iterator_eager_fallback(resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], external_state_policy: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if external_state_policy is None:
        external_state_policy = 0
    external_state_policy = _execute.make_int(external_state_policy, 'external_state_policy')
    resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
    _inputs_flat = [resource_handle]
    _attrs = ('external_state_policy', external_state_policy)
    _result = _execute.execute(b'SerializeIterator', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SerializeIterator', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result