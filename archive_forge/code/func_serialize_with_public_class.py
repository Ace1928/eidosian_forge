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
def serialize_with_public_class(cls, inner_config=None):
    """Serializes classes from public Keras API or object registration.

    Called to check and retrieve the config of any class that has a public
    Keras API or has been registered as serializable via
    `keras.saving.register_keras_serializable()`.
    """
    keras_api_name = tf_export.get_canonical_name_for_symbol(cls, api_name='keras')
    if keras_api_name is None:
        registered_name = object_registration.get_registered_name(cls)
        if registered_name is None:
            return None
        return {'module': cls.__module__, 'class_name': cls.__name__, 'config': inner_config, 'registered_name': registered_name}
    parts = keras_api_name.split('.')
    return {'module': '.'.join(parts[:-1]), 'class_name': parts[-1], 'config': inner_config, 'registered_name': None}