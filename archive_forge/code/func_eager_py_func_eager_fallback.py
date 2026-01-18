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
def eager_py_func_eager_fallback(input, token: str, Tout, is_async: bool, name, ctx):
    token = _execute.make_str(token, 'token')
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'eager_py_func' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if is_async is None:
        is_async = False
    is_async = _execute.make_bool(is_async, 'is_async')
    _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
    _inputs_flat = list(input)
    _attrs = ('token', token, 'is_async', is_async, 'Tin', _attr_Tin, 'Tout', Tout)
    _result = _execute.execute(b'EagerPyFunc', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('EagerPyFunc', _inputs_flat, _attrs, _result)
    return _result