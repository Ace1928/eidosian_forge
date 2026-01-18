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
@tf_export('compat.dimension_at_index', v1=['dimension_at_index', 'compat.dimension_at_index'])
def dimension_at_index(shape, index):
    """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  If you want to retrieve the Dimension instance corresponding to a certain
  index in a TensorShape instance, use this utility, like this:

  ```
  # If you had this in your V1 code:
  dim = tensor_shape[i]

  # Use `dimension_at_index` as direct replacement compatible with both V1 & V2:
  dim = dimension_at_index(tensor_shape, i)

  # Another possibility would be this, but WARNING: it only works if the
  # tensor_shape instance has a defined rank.
  dim = tensor_shape.dims[i]  # `dims` may be None if the rank is undefined!

  # In native V2 code, we recommend instead being more explicit:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]

  # Being more explicit will save you from the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be (as the Dimension object was
  # instantiated on the fly.
  ```

  Args:
    shape: A TensorShape instance.
    index: An integer index.

  Returns:
    A dimension object.
  """
    assert isinstance(shape, TensorShape)
    if shape.rank is None:
        return Dimension(None)
    else:
        return shape.dims[index]