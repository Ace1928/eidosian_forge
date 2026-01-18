from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
def global_avg(self, input_node):
    return tf.math.reduce_mean(input_node, axis=-2)