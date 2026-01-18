import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def get_common_shape(x, y):
    """Find a `TensorShape` that is compatible with both `x` and `y`."""
    if x is None != y is None:
        raise RuntimeError('Cannot find a common shape when LHS shape is None but RHS shape is not (or vice versa): %s vs. %s' % (x, y))
    if x is None:
        return None
    if not isinstance(x, tensor_shape.TensorShape):
        raise TypeError('Expected x to be a TensorShape but saw %s' % (x,))
    if not isinstance(y, tensor_shape.TensorShape):
        raise TypeError('Expected y to be a TensorShape but saw %s' % (y,))
    if x.rank != y.rank or x.rank is None:
        return tensor_shape.TensorShape(None)
    dims = []
    for dim_x, dim_y in zip(x.dims, y.dims):
        if dim_x != dim_y or tensor_shape.dimension_value(dim_x) is None or tensor_shape.dimension_value(dim_y) is None:
            dims.append(None)
        else:
            dims.append(tensor_shape.dimension_value(dim_x))
    return tensor_shape.TensorShape(dims)