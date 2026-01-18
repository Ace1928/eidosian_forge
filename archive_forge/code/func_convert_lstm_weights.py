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
def convert_lstm_weights(weights, from_cudnn=True):
    """Converts the weights between CuDNNLSTM and LSTM.

      Args:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with LSTM.
      """
    kernels = transform_kernels(weights[0], transpose_input(from_cudnn), n_gates)
    recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
    if from_cudnn:
        biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
    else:
        biases = np.tile(0.5 * weights[2], 2)
    return [kernels, recurrent_kernels, biases]