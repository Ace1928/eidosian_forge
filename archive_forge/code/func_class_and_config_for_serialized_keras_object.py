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
def class_and_config_for_serialized_keras_object(config, module_objects=None, custom_objects=None, printable_module_name='object'):
    """Returns the class name and config for a serialized keras object."""
    if not isinstance(config, dict) or 'class_name' not in config or 'config' not in config:
        raise ValueError('Improper config format: ' + str(config))
    class_name = config['class_name']
    cls = get_registered_object(class_name, custom_objects, module_objects)
    if cls is None:
        raise ValueError('Unknown {}: {}. Please ensure this object is passed to the `custom_objects` argument. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.'.format(printable_module_name, class_name))
    cls_config = config['config']
    if isinstance(cls_config, list):
        return (cls, cls_config)
    deserialized_objects = {}
    for key, item in cls_config.items():
        if key == 'name':
            deserialized_objects[key] = item
        elif isinstance(item, dict) and '__passive_serialization__' in item:
            deserialized_objects[key] = deserialize_keras_object(item, module_objects=module_objects, custom_objects=custom_objects, printable_module_name='config_item')
        elif isinstance(item, str) and tf_inspect.isfunction(get_registered_object(item, custom_objects)):
            deserialized_objects[key] = get_registered_object(item, custom_objects)
    for key, item in deserialized_objects.items():
        cls_config[key] = deserialized_objects[key]
    return (cls, cls_config)