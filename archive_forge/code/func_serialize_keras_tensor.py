import copy
import inspect
import typing
import warnings
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import global_state
from keras.src.layers.core.input_layer import Input
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.model import Model
from keras.src.ops.function import Function
from keras.src.ops.function import _build_map
from keras.src.ops.function import make_node_key
from keras.src.ops.node import KerasHistory
from keras.src.ops.node import Node
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils import tree
def serialize_keras_tensor(x):
    if isinstance(x, backend.KerasTensor):
        operation, node_index, tensor_index = x._keras_history
        irrelevant_node_count = 0
        for node in operation._inbound_nodes[:node_index]:
            if node not in own_nodes:
                irrelevant_node_count += 1
        x._keras_history = KerasHistory(operation, node_index - irrelevant_node_count, tensor_index)
        serialized = serialization_lib.serialize_keras_object(x)
        x._keras_history = KerasHistory(operation, node_index, tensor_index)
        return serialized
    return x