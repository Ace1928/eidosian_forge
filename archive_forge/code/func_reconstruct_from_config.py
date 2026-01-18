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
def reconstruct_from_config(config, custom_objects=None, created_layers=None):
    """Reconstructs graph from config object.

  Args:
    config: Dictionary returned from Network.get_config()
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization.
    created_layers: Optional dictionary mapping names to Layer objects. Any
      layer not in this dictionary will be created and added to the dict.
      This function will add new nodes to all layers (excluding InputLayers),
      instead of re-using pre-existing nodes in the layers.

  Returns:
    Tuple of (input tensors, output tensors, dictionary of created layers)
  """
    created_layers = created_layers or collections.OrderedDict()
    node_index_map = {}
    node_count_by_layer = {}
    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
        if layer not in unprocessed_nodes:
            unprocessed_nodes[layer] = [node_data]
        else:
            unprocessed_nodes[layer].append(node_data)

    def get_node_index(layer, config_node_index):
        """Returns node index in layer (might differ from config_node_index)."""
        if isinstance(layer, input_layer_module.InputLayer):
            return 0
        return node_index_map.get((layer.name, config_node_index), None)

    def _deserialize_keras_tensors(kwargs, layer_map):
        """Deserializes Keras Tensors passed to `call`.."""

        def _deserialize_keras_tensor(t):
            """Deserializes a single Keras Tensor passed to `call`."""
            if isinstance(t, tf_utils.ListWrapper):
                t = t.as_list()
                layer_name = t[0]
                node_index = t[1]
                tensor_index = t[2]
                layer = layer_map[layer_name]
                new_node_index = get_node_index(layer, node_index)
                if new_node_index is None:
                    raise IndexError
                node = layer._inbound_nodes[new_node_index]
                return nest.flatten(node.outputs)[tensor_index]
            return t
        kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
        return nest.map_structure(_deserialize_keras_tensor, kwargs)

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

    def process_layer(layer_data):
        """Deserializes a layer, then call it on appropriate inputs.

    Args:
        layer_data: layer config dict.

    Raises:
        ValueError: In case of improperly formatted `layer_data` dict.
    """
        layer_name = layer_data['name']
        if layer_name in created_layers:
            layer = created_layers[layer_name]
        else:
            from tensorflow.python.keras.layers import deserialize as deserialize_layer
            layer = deserialize_layer(layer_data, custom_objects=custom_objects)
            created_layers[layer_name] = layer
        node_count_by_layer[layer] = int(_should_skip_first_node(layer))
        inbound_nodes_data = layer_data['inbound_nodes']
        inbound_nodes_data = tf_utils.convert_inner_node_data(inbound_nodes_data, wrap=True)
        for node_data in inbound_nodes_data:
            add_unprocessed_node(layer, node_data)
    for layer_data in config['layers']:
        process_layer(layer_data)
    while unprocessed_nodes:
        for layer_data in config['layers']:
            layer = created_layers[layer_data['name']]
            if layer in unprocessed_nodes:
                for node_data in unprocessed_nodes.pop(layer):
                    process_node(layer, node_data)
    input_tensors = []
    output_tensors = []
    input_layers = tf_utils.convert_inner_node_data(config['input_layers'], wrap=True)
    for layer_data in nest.flatten(input_layers):
        layer_name, node_index, tensor_index = layer_data.as_list()
        assert layer_name in created_layers
        layer = created_layers[layer_name]
        node_index = get_node_index(layer, node_index)
        layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
        input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])
    output_layers = tf_utils.convert_inner_node_data(config['output_layers'], wrap=True)
    for layer_data in nest.flatten(output_layers):
        layer_name, node_index, tensor_index = layer_data.as_list()
        assert layer_name in created_layers
        layer = created_layers[layer_name]
        node_index = get_node_index(layer, node_index)
        layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
        output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])
    input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
    output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
    return (input_tensors, output_tensors, created_layers)