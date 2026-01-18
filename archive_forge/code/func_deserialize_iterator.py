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
def deserialize_iterator(resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], serialized: _atypes.TensorFuzzingAnnotation[_atypes.Variant], name=None):
    """Converts the given variant tensor to an iterator and stores it in the given resource.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    serialized: A `Tensor` of type `variant`.
      A variant tensor storing the state of the iterator contained in the
      resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DeserializeIterator', name, resource_handle, serialized)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return deserialize_iterator_eager_fallback(resource_handle, serialized, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DeserializeIterator', resource_handle=resource_handle, serialized=serialized, name=name)
    return _op