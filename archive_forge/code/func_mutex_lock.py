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
def mutex_lock(mutex: _atypes.TensorFuzzingAnnotation[_atypes.Resource], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Locks a mutex resource.  The output is the lock.  So long as the lock tensor

  is alive, any other request to use `MutexLock` with this mutex will wait.

  This is particularly useful for creating a critical section when used in
  conjunction with `MutexLockIdentity`:

  ```python

  mutex = mutex_v2(
    shared_name=handle_name, container=container, name=name)

  def execute_in_critical_section(fn, *args, **kwargs):
    lock = gen_resource_variable_ops.mutex_lock(mutex)

    with ops.control_dependencies([lock]):
      r = fn(*args, **kwargs)

    with ops.control_dependencies(nest.flatten(r)):
      with ops.colocate_with(mutex):
        ensure_lock_exists = mutex_lock_identity(lock)

      # Make sure that if any element of r is accessed, all of
      # them are executed together.
      r = nest.map_structure(tf.identity, r)

    with ops.control_dependencies([ensure_lock_exists]):
      return nest.map_structure(tf.identity, r)
  ```

  While `fn` is running in the critical section, no other functions which wish to
  use this critical section may run.

  Often the use case is that two executions of the same graph, in parallel,
  wish to run `fn`; and we wish to ensure that only one of them executes
  at a time.  This is especially important if `fn` modifies one or more
  variables at a time.

  It is also useful if two separate functions must share a resource, but we
  wish to ensure the usage is exclusive.

  Args:
    mutex: A `Tensor` of type `resource`. The mutex resource to lock.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MutexLock', name, mutex)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mutex_lock_eager_fallback(mutex, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MutexLock', mutex=mutex, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('MutexLock', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result