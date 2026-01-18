import functools
import operator
from typing import Optional, Sequence, Type
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export('compat.dimension_value', v1=['dimension_value', 'compat.dimension_value'])
def dimension_value(dimension):
    """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  When accessing the value of a TensorShape dimension,
  use this utility, like this:

  ```
  # If you had this in your V1 code:
  value = tensor_shape[i].value

  # Use `dimension_value` as direct replacement compatible with both V1 & V2:
  value = dimension_value(tensor_shape[i])

  # This would be the V2 equivalent:
  value = tensor_shape[i]  # Warning: this will return the dim value in V2!
  ```

  Args:
    dimension: Either a `Dimension` instance, an integer, or None.

  Returns:
    A plain value, i.e. an integer or None.
  """
    if isinstance(dimension, Dimension):
        return dimension.value
    return dimension