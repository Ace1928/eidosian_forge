from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
def keras_tensor_to_placeholder(x):
    """Construct a graph placeholder to represent a KerasTensor when tracing."""
    if isinstance(x, KerasTensor):
        return x._to_placeholder()
    else:
        return x