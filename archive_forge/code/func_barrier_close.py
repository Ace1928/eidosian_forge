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
def barrier_close(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], cancel_pending_enqueues: bool=False, name=None):
    """Closes the given barrier.

  This operation signals that no more new elements will be inserted in the
  given barrier. Subsequent InsertMany that try to introduce a new key will fail.
  Subsequent InsertMany operations that just add missing components to already
  existing elements will continue to succeed. Subsequent TakeMany operations will
  continue to succeed if sufficient completed elements remain in the barrier.
  Subsequent TakeMany operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the barrier's queue will be canceled. InsertMany will fail, even
      if no new key is introduced.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("barrier_close op does not support eager execution. Arg 'handle' is a ref.")
    if cancel_pending_enqueues is None:
        cancel_pending_enqueues = False
    cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, 'cancel_pending_enqueues')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BarrierClose', handle=handle, cancel_pending_enqueues=cancel_pending_enqueues, name=name)
    return _op