import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def is_builtin_layer(layer):
    if not getattr(layer, '_keras_api_names', None):
        return False
    return layer._keras_api_names != ('keras.layers.Layer',) and layer._keras_api_names_v1 != ('keras.layers.Layer',)