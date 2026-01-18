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
def queue_close_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], cancel_pending_enqueues: bool=False, name=None):
    """Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be canceled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QueueCloseV2', name, handle, 'cancel_pending_enqueues', cancel_pending_enqueues)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return queue_close_v2_eager_fallback(handle, cancel_pending_enqueues=cancel_pending_enqueues, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if cancel_pending_enqueues is None:
        cancel_pending_enqueues = False
    cancel_pending_enqueues = _execute.make_bool(cancel_pending_enqueues, 'cancel_pending_enqueues')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QueueCloseV2', handle=handle, cancel_pending_enqueues=cancel_pending_enqueues, name=name)
    return _op