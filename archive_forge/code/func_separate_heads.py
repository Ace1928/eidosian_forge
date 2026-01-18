from typing import Optional
from typing import Union
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import applications
from tensorflow.keras import layers
from autokeras import keras_layers
from autokeras.blocks import reduction
from autokeras.engine import block as block_module
from autokeras.utils import io_utils
from autokeras.utils import layer_utils
from autokeras.utils import utils
@staticmethod
def separate_heads(x, batch_size, num_heads, projection_dim):
    x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
    return tf.transpose(x, perm=[0, 2, 1, 3])