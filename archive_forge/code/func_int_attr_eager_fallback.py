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
def int_attr_eager_fallback(foo: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    if foo is None:
        foo = 1
    foo = _execute.make_int(foo, 'foo')
    _inputs_flat = []
    _attrs = ('foo', foo)
    _result = _execute.execute(b'IntAttr', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('IntAttr', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result