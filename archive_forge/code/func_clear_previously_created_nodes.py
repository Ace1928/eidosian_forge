import copy
import warnings
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.module import module
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def clear_previously_created_nodes(layer, created_nodes):
    """Remove nodes from `created_nodes` from the layer's inbound_nodes."""
    for node in layer._inbound_nodes:
        prev_layers = node.inbound_layers
        for prev_layer in nest.flatten(prev_layers):
            prev_layer._outbound_nodes = [n for n in prev_layer._outbound_nodes if n not in created_nodes]
    layer._inbound_nodes = [n for n in layer._inbound_nodes if n not in created_nodes]