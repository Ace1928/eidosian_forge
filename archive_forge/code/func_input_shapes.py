import collections
import copy
import json
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
@property
def input_shapes(self):
    input_shapes = nest.map_structure(backend.int_shape, self.input_tensors)
    if len(input_shapes) == 1 and (not self.is_input):
        return input_shapes[0]
    return input_shapes