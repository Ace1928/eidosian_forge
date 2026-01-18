import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
@tf_contextlib.contextmanager
def resource_tracker_scope(resource_tracker):
    """A context to manage resource trackers.

  Use this in order to collect up all resources created within a block of code.
  Example usage:

  ```python
  resource_tracker = ResourceTracker()
  with resource_tracker_scope(resource_tracker):
    resource = TrackableResource()

  assert resource_tracker.resources == [resource]

  Args:
    resource_tracker: The passed in ResourceTracker object

  Yields:
    A scope in which the resource_tracker is active.
  """
    global _RESOURCE_TRACKER_STACK
    old = list(_RESOURCE_TRACKER_STACK)
    _RESOURCE_TRACKER_STACK.append(resource_tracker)
    try:
        yield
    finally:
        _RESOURCE_TRACKER_STACK = old