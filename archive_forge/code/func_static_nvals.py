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
def static_nvals(self):
    """The number of values in this partition, if statically known.

    ```python
    self.value_rowids().shape == [self.static_vals]
    ```

    Returns:
      The number of values in this partition as an `int` (if statically known);
      or `None` (otherwise).
    """
    if self._nvals is not None:
        nvals = tensor_util.constant_value(self._nvals)
        if nvals is not None:
            return nvals
    if self._value_rowids is not None:
        nvals = tensor_shape.dimension_at_index(self._value_rowids.shape, 0)
        if nvals.value is not None:
            return nvals.value
    return None