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
def foo2_eager_fallback(a: _atypes.TensorFuzzingAnnotation[_atypes.Float32], b: _atypes.TensorFuzzingAnnotation[_atypes.String], c: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    a = _ops.convert_to_tensor(a, _dtypes.float32)
    b = _ops.convert_to_tensor(b, _dtypes.string)
    c = _ops.convert_to_tensor(c, _dtypes.string)
    _inputs_flat = [a, b, c]
    _attrs = None
    _result = _execute.execute(b'Foo2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Foo2', _inputs_flat, _attrs, _result)
    _result = _Foo2Output._make(_result)
    return _result