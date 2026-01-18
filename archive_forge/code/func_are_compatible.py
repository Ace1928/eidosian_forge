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
def are_compatible(spec1, spec2):
    """Indicates whether two type specifications are compatible.

  Two type specifications are compatible if they have the same nested structure
  and the their individual components are pair-wise compatible.

  Args:
    spec1: A `tf.TypeSpec` object to compare.
    spec2: A `tf.TypeSpec` object to compare.

  Returns:
    `True` if the two type specifications are compatible and `False` otherwise.
  """
    try:
        nest.assert_same_structure(spec1, spec2)
    except TypeError:
        return False
    except ValueError:
        return False
    for s1, s2 in zip(nest.flatten(spec1), nest.flatten(spec2)):
        if not s1.is_compatible_with(s2) or not s2.is_compatible_with(s1):
            return False
    return True