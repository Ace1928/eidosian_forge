import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
    """A context that disables automatic dependency tracking when assigning attrs.

  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.

  Example usage:
  ```
  model = tf.keras.Model()
  model.arr1 = []  # Creates a ListWrapper object
  with no_automatic_dependency_tracking_scope(model):
    model.arr2 = []  # Creates a regular, untracked python list
  ```

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies.
  """
    previous_value = getattr(obj, '_setattr_tracking', True)
    obj._setattr_tracking = False
    try:
        yield
    finally:
        obj._setattr_tracking = previous_value