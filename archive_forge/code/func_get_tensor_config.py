import copy
import inspect
import warnings
import tree
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import global_state
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.model import Model
from keras.src.ops.function import Function
from keras.src.ops.function import make_node_key
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def get_tensor_config(tensor):
    operation = tensor._keras_history[0]
    node_index = tensor._keras_history[1]
    tensor_index = tensor._keras_history[2]
    node_key = make_node_key(operation, node_index)
    assert node_key in self._nodes
    new_node_index = node_reindexing_map[node_key]
    return [operation.name, new_node_index, tensor_index]