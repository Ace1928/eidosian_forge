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
def get_element_at_index_eager_fallback(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, name, ctx):
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'get_element_at_index' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'get_element_at_index' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
    index = _ops.convert_to_tensor(index, _dtypes.int64)
    _inputs_flat = [dataset, index]
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'GetElementAtIndex', len(output_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('GetElementAtIndex', _inputs_flat, _attrs, _result)
    return _result