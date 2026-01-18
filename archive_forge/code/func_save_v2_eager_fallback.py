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
def save_v2_eager_fallback(prefix: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_names: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slices: _atypes.TensorFuzzingAnnotation[_atypes.String], tensors, name, ctx):
    _attr_dtypes, tensors = _execute.convert_to_mixed_eager_tensors(tensors, ctx)
    prefix = _ops.convert_to_tensor(prefix, _dtypes.string)
    tensor_names = _ops.convert_to_tensor(tensor_names, _dtypes.string)
    shape_and_slices = _ops.convert_to_tensor(shape_and_slices, _dtypes.string)
    _inputs_flat = [prefix, tensor_names, shape_and_slices] + list(tensors)
    _attrs = ('dtypes', _attr_dtypes)
    _result = _execute.execute(b'SaveV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result