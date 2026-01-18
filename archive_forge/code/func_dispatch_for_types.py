import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def dispatch_for_types(op, *types):
    """Decorator to declare that a Python function overrides an op for a type.

  The decorated function is used to override `op` if any of the arguments or
  keyword arguments (including elements of lists or tuples) have one of the
  specified types.

  Example:

  ```python
  @dispatch_for_types(math_ops.add, RaggedTensor, RaggedTensorValue)
  def ragged_add(x, y, name=None): ...
  ```

  Args:
    op: Python function: the operation that should be overridden.
    *types: The argument types for which this function should be used.
  """

    def decorator(func):
        _TypeBasedDispatcher(get_compatible_func(op, func), types).register(op)
        return func
    return decorator