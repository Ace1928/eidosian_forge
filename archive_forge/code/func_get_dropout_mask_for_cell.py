import collections
import warnings
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def get_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the dropout mask for RNN cell's input.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
        return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)