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
def d_tensor_restore_v2_eager_fallback(prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], input_shapes, input_layouts, dtypes, name, ctx):
    if not isinstance(input_shapes, (list, tuple)):
        raise TypeError("Expected list for 'input_shapes' argument to 'd_tensor_restore_v2' Op, not %r." % input_shapes)
    input_shapes = [_execute.make_shape(_s, 'input_shapes') for _s in input_shapes]
    if not isinstance(input_layouts, (list, tuple)):
        raise TypeError("Expected list for 'input_layouts' argument to 'd_tensor_restore_v2' Op, not %r." % input_layouts)
    input_layouts = [_execute.make_str(_s, 'input_layouts') for _s in input_layouts]
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'd_tensor_restore_v2' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    prefix = _ops.convert_to_tensor(prefix, _dtypes.string)
    tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
    shape_and_slices = _ops.convert_to_tensor(shape_and_slices, _dtypes.string)
    _inputs_flat = [prefix, tensor_names, shape_and_slices]
    _attrs = ('input_shapes', input_shapes, 'input_layouts', input_layouts, 'dtypes', dtypes)
    _result = _execute.execute(b'DTensorRestoreV2', len(dtypes), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DTensorRestoreV2', _inputs_flat, _attrs, _result)
    return _result