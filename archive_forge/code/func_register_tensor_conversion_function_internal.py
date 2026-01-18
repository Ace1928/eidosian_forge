import collections
import threading
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
def register_tensor_conversion_function_internal(base_type, conversion_func, priority=100):
    """Internal version of register_tensor_conversion_function.

  See docstring of `register_tensor_conversion_function` for details.

  The internal version of the function allows registering conversions
  for types in the _UNCONVERTIBLE_TYPES tuple.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
    base_types = base_type if isinstance(base_type, tuple) else (base_type,)
    if any((not isinstance(x, type) for x in base_types)):
        raise TypeError(f'Argument `base_type` must be a type or a tuple of types. Obtained: {base_type}')
    del base_types
    if not callable(conversion_func):
        raise TypeError(f'Argument `conversion_func` must be callable. Received {conversion_func}.')
    with _tensor_conversion_func_lock:
        _tensor_conversion_func_registry[priority].append((base_type, conversion_func))
        _tensor_conversion_func_cache.clear()