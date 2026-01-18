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
def rpc_client_eager_fallback(server_address: _atypes.TensorFuzzingAnnotation[_atypes.String], timeout_in_ms: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shared_name: str, list_registered_methods: bool, name, ctx):
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if list_registered_methods is None:
        list_registered_methods = False
    list_registered_methods = _execute.make_bool(list_registered_methods, 'list_registered_methods')
    server_address = _ops.convert_to_tensor(server_address, _dtypes.string)
    timeout_in_ms = _ops.convert_to_tensor(timeout_in_ms, _dtypes.int64)
    _inputs_flat = [server_address, timeout_in_ms]
    _attrs = ('shared_name', shared_name, 'list_registered_methods', list_registered_methods)
    _result = _execute.execute(b'RpcClient', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RpcClient', _inputs_flat, _attrs, _result)
    _result = _RpcClientOutput._make(_result)
    return _result