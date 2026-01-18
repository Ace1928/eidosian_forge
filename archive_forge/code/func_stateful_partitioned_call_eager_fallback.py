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
def stateful_partitioned_call_eager_fallback(args, Tout, f, config: str, config_proto: str, executor_type: str, name, ctx):
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'stateful_partitioned_call' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if config is None:
        config = ''
    config = _execute.make_str(config, 'config')
    if config_proto is None:
        config_proto = ''
    config_proto = _execute.make_str(config_proto, 'config_proto')
    if executor_type is None:
        executor_type = ''
    executor_type = _execute.make_str(executor_type, 'executor_type')
    _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, ctx)
    _inputs_flat = list(args)
    _attrs = ('Tin', _attr_Tin, 'Tout', Tout, 'f', f, 'config', config, 'config_proto', config_proto, 'executor_type', executor_type)
    _result = _execute.execute(b'StatefulPartitionedCall', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatefulPartitionedCall', _inputs_flat, _attrs, _result)
    return _result