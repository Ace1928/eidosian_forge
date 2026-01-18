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
def destroy_resource_op(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], ignore_lookup_error: bool=True, name=None):
    """Deletes the resource specified by the handle.

  All subsequent operations using the resource will result in a NotFound
  error status.

  Args:
    resource: A `Tensor` of type `resource`. handle to the resource to delete.
    ignore_lookup_error: An optional `bool`. Defaults to `True`.
      whether to ignore the error when the resource
      doesn't exist.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DestroyResourceOp', name, resource, 'ignore_lookup_error', ignore_lookup_error)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return destroy_resource_op_eager_fallback(resource, ignore_lookup_error=ignore_lookup_error, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if ignore_lookup_error is None:
        ignore_lookup_error = True
    ignore_lookup_error = _execute.make_bool(ignore_lookup_error, 'ignore_lookup_error')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DestroyResourceOp', resource=resource, ignore_lookup_error=ignore_lookup_error, name=name)
    return _op