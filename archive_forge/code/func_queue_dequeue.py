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
def queue_dequeue(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], component_types, timeout_ms: int=-1, name=None):
    """Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("queue_dequeue op does not support eager execution. Arg 'handle' is a ref.")
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'queue_dequeue' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if timeout_ms is None:
        timeout_ms = -1
    timeout_ms = _execute.make_int(timeout_ms, 'timeout_ms')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QueueDequeue', handle=handle, component_types=component_types, timeout_ms=timeout_ms, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('component_types', _op.get_attr('component_types'), 'timeout_ms', _op._get_attr_int('timeout_ms'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QueueDequeue', _inputs_flat, _attrs, _result)
    return _result