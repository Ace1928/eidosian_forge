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
def concat_offset_eager_fallback(concat_dim: _atypes.TensorFuzzingAnnotation[_atypes.Int32], shape: List[_atypes.TensorFuzzingAnnotation[TV_ConcatOffset_shape_type]], name, ctx):
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Expected list for 'shape' argument to 'concat_offset' Op, not %r." % shape)
    _attr_N = len(shape)
    _attr_shape_type, shape = _execute.args_to_matching_eager(list(shape), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    concat_dim = _ops.convert_to_tensor(concat_dim, _dtypes.int32)
    _inputs_flat = [concat_dim] + list(shape)
    _attrs = ('N', _attr_N, 'shape_type', _attr_shape_type)
    _result = _execute.execute(b'ConcatOffset', _attr_N, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ConcatOffset', _inputs_flat, _attrs, _result)
    return _result