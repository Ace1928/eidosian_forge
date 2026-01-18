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
def convert_gru_weights(weights, from_cudnn=True):
    """Converts the weights between CuDNNGRU and GRU.

      Args:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with GRU.
      """
    kernels = transform_kernels(weights[0], transpose_input(from_cudnn), n_gates)
    recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
    biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
    return [kernels, recurrent_kernels, biases]