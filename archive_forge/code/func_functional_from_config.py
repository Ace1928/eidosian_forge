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
def functional_from_config(cls, config, custom_objects=None):
    """Instantiates a Functional model from its config (from `get_config()`).

    Args:
        cls: Class of the model, e.g. a custom subclass of `Model`.
        config: Output of `get_config()` for the original model instance.
        custom_objects: Optional dict of custom objects.

    Returns:
        An instance of `cls`.
    """
    created_layers = {}
    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
        """Add node to layer list

        Arg:
            layer: layer object
            node_data: Node data specifying layer call
        """
        if layer not in unprocessed_nodes:
            unprocessed_nodes[layer] = [node_data]
        else:
            unprocessed_nodes[layer].append(node_data)

    def process_node(layer, node_data):
        """Reconstruct node by linking to inbound layers

        Args:
            layer: Layer to process
            node_data: List of layer configs
        """
        args, kwargs = deserialize_node(node_data, created_layers)
        layer(*args, **kwargs)

    def process_layer(layer_data):
        """Deserializes a layer, then call it on appropriate inputs.

        Args:
            layer_data: layer config dict.
        """
        layer_name = layer_data['name']
        if 'module' not in layer_data:
            layer = saving_utils.model_from_config(layer_data, custom_objects=custom_objects)
        else:
            layer = serialization_lib.deserialize_keras_object(layer_data, custom_objects=custom_objects)
        created_layers[layer_name] = layer
        inbound_nodes_data = layer_data['inbound_nodes']
        for node_data in inbound_nodes_data:
            add_unprocessed_node(layer, node_data)
    for layer_data in config['layers']:
        process_layer(layer_data)
    while unprocessed_nodes:
        for layer_data in config['layers']:
            layer = created_layers[layer_data['name']]
            if layer in unprocessed_nodes:
                node_data_list = unprocessed_nodes[layer]
                node_index = 0
                while node_index < len(node_data_list):
                    node_data = node_data_list[node_index]
                    try:
                        process_node(layer, node_data)
                    except IndexError:
                        break
                    node_index += 1
                if node_index < len(node_data_list):
                    unprocessed_nodes[layer] = node_data_list[node_index:]
                else:
                    del unprocessed_nodes[layer]
    name = config.get('name')
    trainable = config.get('trainable')

    def get_tensor(layer_name, node_index, tensor_index):
        assert layer_name in created_layers
        layer = created_layers[layer_name]
        layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
        return layer_output_tensors[tensor_index]

    def map_tensors(tensors):
        if isinstance(tensors, dict):
            return {k: get_tensor(*v) for k, v in tensors.items()}
        else:
            return [get_tensor(*v) for v in tensors]
    input_tensors = map_tensors(config['input_layers'])
    output_tensors = map_tensors(config['output_layers'])
    return cls(inputs=input_tensors, outputs=output_tensors, name=name, trainable=trainable)