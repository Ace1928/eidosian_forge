import json
import os
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.saving import model_config as model_config_lib
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
def convert_nested_model(weights):
    """Converts layers nested in `Model` or `Sequential`.

    This function uses `preprocess_weights_for_loading()` for converting nested
    layers.

    Args:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    trainable_weights = weights[:len(layer.trainable_weights)]
    non_trainable_weights = weights[len(layer.trainable_weights):]
    new_trainable_weights = []
    new_non_trainable_weights = []
    for sublayer in layer.layers:
        num_trainable_weights = len(sublayer.trainable_weights)
        num_non_trainable_weights = len(sublayer.non_trainable_weights)
        if sublayer.weights:
            preprocessed = preprocess_weights_for_loading(layer=sublayer, weights=trainable_weights[:num_trainable_weights] + non_trainable_weights[:num_non_trainable_weights], original_keras_version=original_keras_version, original_backend=original_backend)
            new_trainable_weights.extend(preprocessed[:num_trainable_weights])
            new_non_trainable_weights.extend(preprocessed[num_trainable_weights:])
            trainable_weights = trainable_weights[num_trainable_weights:]
            non_trainable_weights = non_trainable_weights[num_non_trainable_weights:]
    return new_trainable_weights + new_non_trainable_weights