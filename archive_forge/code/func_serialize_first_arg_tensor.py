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
def serialize_first_arg_tensor(t):
    if is_keras_tensor(t):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        data = [kh.layer.name, new_node_index, kh.tensor_index, kwargs]
    else:
        data = [_CONSTANT_VALUE, -1, _serialize_keras_tensor(t), kwargs]
    return tf_utils.ListWrapper(data)