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
def out_type_list_restrict_eager_fallback(t, name, ctx):
    if not isinstance(t, (list, tuple)):
        raise TypeError("Expected list for 't' argument to 'out_type_list_restrict' Op, not %r." % t)
    t = [_execute.make_type(_t, 't') for _t in t]
    _inputs_flat = []
    _attrs = ('t', t)
    _result = _execute.execute(b'OutTypeListRestrict', len(t), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('OutTypeListRestrict', _inputs_flat, _attrs, _result)
    return _result