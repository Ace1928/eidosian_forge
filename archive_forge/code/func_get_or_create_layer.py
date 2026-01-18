from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import functools
import tensorflow.compat.v2 as tf
from keras.src.engine import base_layer
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def get_or_create_layer(self, name, create_layer_method):
    if name not in self._layers:
        layer = create_layer_method()
        self._layers[name] = layer
        if isinstance(layer, base_layer.Layer):
            self._regularizers[name] = lambda: tf.math.reduce_sum(layer.losses)
    return self._layers[name]