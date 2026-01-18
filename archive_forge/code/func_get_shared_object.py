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
def get_shared_object(obj_id):
    """Retrieve an object previously seen during deserialization."""
    if getattr(SHARED_OBJECTS, 'enabled', False):
        return SHARED_OBJECTS.id_to_obj_map.get(obj_id, None)