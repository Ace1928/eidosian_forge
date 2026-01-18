import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def mean_update():
    """Update self.moving_mean with the most recent data point."""
    if use_fused_avg_updates:
        if input_batch_size is not None:
            new_mean = control_flow_util.smart_cond(input_batch_size > 0, lambda: mean, lambda: self.moving_mean)
        else:
            new_mean = mean
        return self._assign_new_value(self.moving_mean, new_mean)
    else:
        return self._assign_moving_average(self.moving_mean, mean, momentum, input_batch_size)