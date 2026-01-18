from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def get_op_name(tensor_name):
    """Extract the Op name from a Tensor name.

  The Op name is everything before a colon, if present,
  not including any ^ prefix denoting a control dependency.

  Args:
    tensor_name: the full name of a Tensor in the graph.
  Returns:
    The name of the Op of which the given Tensor is an output.
  Raises:
    ValueError: if tensor_name is None or empty.
  """
    if not tensor_name:
        raise ValueError(f'Tensor name cannot be empty or None. Received: {tensor_name}.')
    if tensor_name.startswith('^'):
        tensor_name = tensor_name[1:]
    if ':' in tensor_name:
        op_name, _ = tensor_name.split(':')
        return op_name
    return tensor_name