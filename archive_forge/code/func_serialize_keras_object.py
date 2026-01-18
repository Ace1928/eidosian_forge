import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def serialize_keras_object(instance):
    """Serialize a Keras object into a JSON-compatible representation.

  Calls to `serialize_keras_object` while underneath the
  `SharedObjectSavingScope` context manager will cause any objects re-used
  across multiple layers to be saved with a special shared object ID. This
  allows the network to be re-created properly during deserialization.

  Args:
    instance: The object to serialize.

  Returns:
    A dict-like, JSON-compatible representation of the object's config.
  """
    _, instance = tf_decorator.unwrap(instance)
    if instance is None:
        return None
    supports_masking = getattr(instance, 'supports_masking', False) or (hasattr(instance, 'compute_mask') and (not is_default(instance.compute_mask)))
    if supports_masking and is_default(instance.get_config):
        warnings.warn('Custom mask layers require a config and must override get_config. When loading, the custom mask layer must be passed to the custom_objects argument.', category=CustomMaskWarning)
    if hasattr(instance, 'get_config'):
        name = get_registered_name(instance.__class__)
        try:
            config = instance.get_config()
        except NotImplementedError as e:
            if _SKIP_FAILED_SERIALIZATION:
                return serialize_keras_class_and_config(name, {_LAYER_UNDEFINED_CONFIG_KEY: True})
            raise e
        serialization_config = {}
        for key, item in config.items():
            if isinstance(item, str):
                serialization_config[key] = item
                continue
            try:
                serialized_item = serialize_keras_object(item)
                if isinstance(serialized_item, dict) and (not isinstance(item, dict)):
                    serialized_item['__passive_serialization__'] = True
                serialization_config[key] = serialized_item
            except ValueError:
                serialization_config[key] = item
        name = get_registered_name(instance.__class__)
        return serialize_keras_class_and_config(name, serialization_config, instance)
    if hasattr(instance, '__name__'):
        return get_registered_name(instance)
    raise ValueError('Cannot serialize', instance)