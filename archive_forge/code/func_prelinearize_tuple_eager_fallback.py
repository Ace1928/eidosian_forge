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
def prelinearize_tuple_eager_fallback(inputs, shapes, layouts, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'prelinearize_tuple' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if layouts is None:
        layouts = []
    if not isinstance(layouts, (list, tuple)):
        raise TypeError("Expected list for 'layouts' argument to 'prelinearize_tuple' Op, not %r." % layouts)
    layouts = [_execute.make_int(_i, 'layouts') for _i in layouts]
    _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
    _inputs_flat = list(inputs)
    _attrs = ('dtypes', _attr_dtypes, 'shapes', shapes, 'layouts', layouts)
    _result = _execute.execute(b'PrelinearizeTuple', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('PrelinearizeTuple', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result