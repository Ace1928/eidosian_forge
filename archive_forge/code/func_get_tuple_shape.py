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
def get_tuple_shape(nb_channels):
    result = list(state_shape)
    if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
    elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
    else:
        raise KeyError
    return tuple(result)