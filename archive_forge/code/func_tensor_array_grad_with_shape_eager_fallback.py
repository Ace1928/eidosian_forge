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
def tensor_array_grad_with_shape_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], shape_to_prepend: _atypes.TensorFuzzingAnnotation[_atypes.Int32], source: str, name, ctx):
    source = _execute.make_str(source, 'source')
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    flow_in = _ops.convert_to_tensor(flow_in, _dtypes.float32)
    shape_to_prepend = _ops.convert_to_tensor(shape_to_prepend, _dtypes.int32)
    _inputs_flat = [handle, flow_in, shape_to_prepend]
    _attrs = ('source', source)
    _result = _execute.execute(b'TensorArrayGradWithShape', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorArrayGradWithShape', _inputs_flat, _attrs, _result)
    _result = _TensorArrayGradWithShapeOutput._make(_result)
    return _result