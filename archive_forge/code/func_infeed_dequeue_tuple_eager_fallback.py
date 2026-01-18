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
def infeed_dequeue_tuple_eager_fallback(dtypes, shapes, name, ctx):
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'infeed_dequeue_tuple' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'infeed_dequeue_tuple' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    _inputs_flat = []
    _attrs = ('dtypes', dtypes, 'shapes', shapes)
    _result = _execute.execute(b'InfeedDequeueTuple', len(dtypes), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('InfeedDequeueTuple', _inputs_flat, _attrs, _result)
    return _result