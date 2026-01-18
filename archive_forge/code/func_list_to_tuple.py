import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def list_to_tuple(maybe_list):
    """Datasets will stack the list of tensor, so switch them to tuples."""
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list