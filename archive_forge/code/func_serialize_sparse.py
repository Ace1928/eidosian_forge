import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['io.serialize_sparse', 'serialize_sparse'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('serialize_sparse')
def serialize_sparse(sp_input, name=None, out_type=dtypes.string):
    """Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).
    out_type: The `dtype` to use for serialization.

  Returns:
    A 3-vector (1-D `Tensor`), with each column representing the serialized
    `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
    return serialize_sparse_v2(sp_input, out_type, name)