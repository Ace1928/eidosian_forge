import importlib
import inspect
import threading
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.saving import object_registration
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from keras.src.utils import generic_utils
from tensorflow.python.util import tf_export
from tensorflow.python.util.tf_export import keras_export
def serialize_with_public_fn(fn, config, fn_module_name=None):
    """Serializes functions from public Keras API or object registration.

    Called to check and retrieve the config of any function that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`. If function's module name is
    already known, returns corresponding config.
    """
    if fn_module_name:
        return {'module': fn_module_name, 'class_name': 'function', 'config': config, 'registered_name': config}
    keras_api_name = tf_export.get_canonical_name_for_symbol(fn, api_name='keras')
    if keras_api_name:
        parts = keras_api_name.split('.')
        return {'module': '.'.join(parts[:-1]), 'class_name': 'function', 'config': config, 'registered_name': config}
    else:
        registered_name = object_registration.get_registered_name(fn)
        if not registered_name and (not fn.__module__ == 'builtins'):
            return None
        return {'module': fn.__module__, 'class_name': 'function', 'config': config, 'registered_name': registered_name}