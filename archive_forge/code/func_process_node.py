import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def process_node(layer, node_data):
    """Deserialize a node.

    Args:
        layer: layer instance.
        node_data: Nested structure of `ListWrapper`.

    Raises:
        ValueError: In case of improperly formatted `node_data`.
    """
    input_tensors = []
    for input_data in nest.flatten(node_data):
        input_data = input_data.as_list()
        inbound_layer_name = input_data[0]
        inbound_node_index = input_data[1]
        inbound_tensor_index = input_data[2]
        if len(input_data) == 3:
            kwargs = {}
        elif len(input_data) == 4:
            kwargs = input_data[3]
            try:
                kwargs = _deserialize_keras_tensors(kwargs, created_layers)
            except IndexError:
                add_unprocessed_node(layer, node_data)
                return
        else:
            raise ValueError('Improperly formatted model config.')
        if inbound_layer_name != node_module._CONSTANT_VALUE:
            inbound_layer = created_layers[inbound_layer_name]
            inbound_node_index = get_node_index(inbound_layer, inbound_node_index)
            if inbound_node_index is None:
                add_unprocessed_node(layer, node_data)
                return
            inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
            input_tensors.append(nest.flatten(inbound_node.outputs)[inbound_tensor_index])
        else:
            input_tensors.append(inbound_tensor_index)
    input_tensors = nest.pack_sequence_as(node_data, input_tensors)
    if input_tensors is not None:
        if not layer._preserve_input_structure_in_config:
            input_tensors = base_layer_utils.unnest_if_single_tensor(input_tensors)
        output_tensors = layer(input_tensors, **kwargs)
        output_index = nest.flatten(output_tensors)[0]._keras_history.node_index
        node_index_map[layer.name, node_count_by_layer[layer]] = output_index
        node_count_by_layer[layer] += 1