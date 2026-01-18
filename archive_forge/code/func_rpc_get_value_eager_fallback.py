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
def rpc_get_value_eager_fallback(status_or: _atypes.TensorFuzzingAnnotation[_atypes.Resource], Tout, name, ctx):
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'rpc_get_value' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    status_or = _ops.convert_to_tensor(status_or, _dtypes.resource)
    _inputs_flat = [status_or]
    _attrs = ('Tout', Tout)
    _result = _execute.execute(b'RpcGetValue', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RpcGetValue', _inputs_flat, _attrs, _result)
    return _result