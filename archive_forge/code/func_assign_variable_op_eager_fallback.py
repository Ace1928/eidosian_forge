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
def assign_variable_op_eager_fallback(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], value: _atypes.TensorFuzzingAnnotation[TV_AssignVariableOp_dtype], validate_shape: bool, name, ctx):
    if validate_shape is None:
        validate_shape = False
    validate_shape = _execute.make_bool(validate_shape, 'validate_shape')
    _attr_dtype, (value,) = _execute.args_to_matching_eager([value], ctx, [])
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, value]
    _attrs = ('dtype', _attr_dtype, 'validate_shape', validate_shape)
    _result = _execute.execute(b'AssignVariableOp', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result