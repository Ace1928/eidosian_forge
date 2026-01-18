import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def get_static_batch_size(layer):
    """Gets the static batch size of a Layer.

  Args:
    layer: a `Layer` instance.

  Returns:
    The static batch size of a Layer.
  """
    batch_input_shape, _ = get_input_shape_and_dtype(layer)
    if batch_input_shape is not None:
        return tensor_shape.Dimension(batch_input_shape[0]).value
    return None