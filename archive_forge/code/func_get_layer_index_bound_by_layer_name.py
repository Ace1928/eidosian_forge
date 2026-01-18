import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def get_layer_index_bound_by_layer_name(model, layer_range=None):
    """Get the layer indexes from the model based on layer names.

    The layer indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `tf.keras.Model` instance.
        layer_names: a list or tuple of 2 strings, the starting layer name and
            ending layer name (both inclusive) for the result. All layers will
            be included when `None` is provided.

    Returns:
        The index value of layer based on its unique name (layer_names).
        Output will be [first_layer_index, last_layer_index + 1].
    """
    if layer_range is not None:
        if len(layer_range) != 2:
            raise ValueError(f'layer_range must be a list or tuple of length 2. Received: layer_range = {layer_range} of length {len(layer_range)}')
        if not isinstance(layer_range[0], str) or not isinstance(layer_range[1], str):
            raise ValueError(f'layer_range should contain string type only. Received: {layer_range}')
    else:
        return [0, len(model.layers)]
    lower_index = [idx for idx, layer in enumerate(model.layers) if re.match(layer_range[0], layer.name)]
    upper_index = [idx for idx, layer in enumerate(model.layers) if re.match(layer_range[1], layer.name)]
    if not lower_index or not upper_index:
        raise ValueError(f'Passed layer_names do not match the layer names in the model. Received: {layer_range}')
    if min(lower_index) > max(upper_index):
        return [min(upper_index), max(lower_index) + 1]
    return [min(lower_index), max(upper_index) + 1]