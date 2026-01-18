import collections
import enum
import json
import numpy as np
import wrapt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
def get_json_type(obj):
    """Serializes any object to a JSON-serializable structure.

  Args:
      obj: the object to serialize

  Returns:
      JSON-serializable structure representing `obj`.

  Raises:
      TypeError: if `obj` cannot be serialized.
  """
    if hasattr(obj, 'get_config'):
        return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    if callable(obj):
        return obj.__name__
    if type(obj).__name__ == type.__name__:
        return obj.__name__
    if isinstance(obj, tensor_shape.Dimension):
        return obj.value
    if isinstance(obj, tensor_shape.TensorShape):
        return obj.as_list()
    if isinstance(obj, dtypes.DType):
        return obj.name
    if isinstance(obj, collections.abc.Mapping):
        return dict(obj)
    if obj is Ellipsis:
        return {'class_name': '__ellipsis__'}
    if isinstance(obj, wrapt.ObjectProxy):
        return obj.__wrapped__
    if isinstance(obj, internal.TypeSpec):
        try:
            type_spec_name = type_spec_registry.get_name(type(obj))
            return {'class_name': 'TypeSpec', 'type_spec': type_spec_name, 'serialized': obj._serialize()}
        except ValueError:
            raise ValueError('Unable to serialize {} to JSON, because the TypeSpec class {} has not been registered.'.format(obj, type(obj)))
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError('Not JSON Serializable:', obj)