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
def assign_variable_op(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], value: _atypes.TensorFuzzingAnnotation[TV_AssignVariableOp_dtype], validate_shape: bool=False, name=None):
    """Assigns a new value to a variable.

  Any ReadVariableOp with a control dependency on this op is guaranteed to return
  this value or a subsequent newer value of the variable.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    value: A `Tensor`. the value to set the new tensor to use.
    validate_shape: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AssignVariableOp', name, resource, value, 'validate_shape', validate_shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return assign_variable_op_eager_fallback(resource, value, validate_shape=validate_shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if validate_shape is None:
        validate_shape = False
    validate_shape = _execute.make_bool(validate_shape, 'validate_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AssignVariableOp', resource=resource, value=value, validate_shape=validate_shape, name=name)
    return _op