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
def register_keras_serializable(package='Custom', name=None):
    """Registers an object with the Keras serialization framework.

  This decorator injects the decorated class or function into the Keras custom
  object dictionary, so that it can be serialized and deserialized without
  needing an entry in the user-provided custom object dict. It also injects a
  function that Keras will call to get the object's serializable string key.

  Note that to be serialized and deserialized, classes must implement the
  `get_config()` method. Functions do not have this requirement.

  The object will be registered under the key 'package>name' where `name`,
  defaults to the object name if not passed.

  Args:
    package: The package that this class belongs to.
    name: The name to serialize this class under in this package. If None, the
      class' name will be used.

  Returns:
    A decorator that registers the decorated class with the passed names.
  """

    def decorator(arg):
        """Registers a class with the Keras serialization framework."""
        class_name = name if name is not None else arg.__name__
        registered_name = package + '>' + class_name
        if tf_inspect.isclass(arg) and (not hasattr(arg, 'get_config')):
            raise ValueError('Cannot register a class that does not have a get_config() method.')
        if registered_name in _GLOBAL_CUSTOM_OBJECTS:
            raise ValueError('%s has already been registered to %s' % (registered_name, _GLOBAL_CUSTOM_OBJECTS[registered_name]))
        if arg in _GLOBAL_CUSTOM_NAMES:
            raise ValueError('%s has already been registered to %s' % (arg, _GLOBAL_CUSTOM_NAMES[arg]))
        _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        _GLOBAL_CUSTOM_NAMES[arg] = registered_name
        return arg
    return decorator