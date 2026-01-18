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
def priority_queue_v2(shapes, component_types=[], capacity: int=-1, container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """A queue that produces elements sorted by the first component value.

  Note that the PriorityQueue requires the first component of any element
  to be a scalar int64, in addition to the other elements declared by
  component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
  and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
  entry in their input (resp. output) lists.

  Args:
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    component_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type of each component in a value.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'PriorityQueueV2', name, 'component_types', component_types, 'shapes', shapes, 'capacity', capacity, 'container', container, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return priority_queue_v2_eager_fallback(component_types=component_types, shapes=shapes, capacity=capacity, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'priority_queue_v2' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if component_types is None:
        component_types = []
    if not isinstance(component_types, (list, tuple)):
        raise TypeError("Expected list for 'component_types' argument to 'priority_queue_v2' Op, not %r." % component_types)
    component_types = [_execute.make_type(_t, 'component_types') for _t in component_types]
    if capacity is None:
        capacity = -1
    capacity = _execute.make_int(capacity, 'capacity')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('PriorityQueueV2', shapes=shapes, component_types=component_types, capacity=capacity, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('component_types', _op.get_attr('component_types'), 'shapes', _op.get_attr('shapes'), 'capacity', _op._get_attr_int('capacity'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('PriorityQueueV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result