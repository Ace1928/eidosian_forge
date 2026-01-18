import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
@property
def static_uniform_row_length(self):
    """The number of values in each row of this partition, if statically known.

    Returns:
      The number of values in each row of this partition as an `int` (if
      statically known); or `None` (otherwise).
    """
    if self._uniform_row_length is not None:
        return tensor_util.constant_value(self._uniform_row_length)
    return None