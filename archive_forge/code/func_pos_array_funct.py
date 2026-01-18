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
def pos_array_funct(maxlen, batch_size):
    pos_ones = tf.ones((batch_size, 1), dtype=tf.int32)
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = tf.expand_dims(positions, 0)
    positions = tf.matmul(pos_ones, positions)
    return positions