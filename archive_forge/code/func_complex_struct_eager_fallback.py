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
def complex_struct_eager_fallback(n_a: int, n_b: int, t_c, name, ctx):
    n_a = _execute.make_int(n_a, 'n_a')
    n_b = _execute.make_int(n_b, 'n_b')
    if not isinstance(t_c, (list, tuple)):
        raise TypeError("Expected list for 't_c' argument to 'complex_struct' Op, not %r." % t_c)
    t_c = [_execute.make_type(_t, 't_c') for _t in t_c]
    _inputs_flat = []
    _attrs = ('n_a', n_a, 'n_b', n_b, 't_c', t_c)
    _result = _execute.execute(b'ComplexStruct', n_a + n_b + len(t_c), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ComplexStruct', _inputs_flat, _attrs, _result)
    _result = [_result[:n_a]] + _result[n_a:]
    _result = _result[:1] + [_result[1:1 + n_b]] + _result[1 + n_b:]
    _result = _result[:2] + [_result[2:]]
    _result = _ComplexStructOutput._make(_result)
    return _result