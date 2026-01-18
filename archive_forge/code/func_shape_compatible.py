from typing import Optional
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils
from autokeras.utils import utils
def shape_compatible(shape1, shape2):
    if len(shape1) != len(shape2):
        return False
    return shape1[:-1] == shape2[:-1]