from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
def is_v1_layer_or_model(obj):
    return isinstance(obj, (base_layer_v1.Layer, training_v1.Model))