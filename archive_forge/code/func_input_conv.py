import numpy as np
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
def input_conv(self, x, w, b=None, padding='valid'):
    conv_out = backend.conv2d(x, w, strides=self.strides, padding=padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
    if b is not None:
        conv_out = backend.bias_add(conv_out, b, data_format=self.data_format)
    return conv_out