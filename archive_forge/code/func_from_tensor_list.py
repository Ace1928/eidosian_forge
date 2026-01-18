import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
def from_tensor_list(element_spec, tensor_list):
    """Returns an element constructed from the given spec and tensor list.

  Args:
    element_spec: A nested structure of `tf.TypeSpec` objects representing to
      element type specification.
    tensor_list: A list of tensors to use for constructing the value.

  Returns:
    An element constructed from the given spec and tensor list.

  Raises:
    ValueError: If the number of tensors needed to construct an element for
      the given spec does not match the given number of tensors or the given
      spec is not compatible with the tensor list.
  """
    return _from_tensor_list_helper(lambda spec, value: spec._from_tensor_list(value), element_spec, tensor_list)