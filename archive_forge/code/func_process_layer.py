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