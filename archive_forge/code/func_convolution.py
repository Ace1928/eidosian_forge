from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
    height_pad = width_pad = (self.pad_value, self.pad_value)
    hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
    hidden_state = self.conv(hidden_state)
    return hidden_state