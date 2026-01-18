import abc
import contextlib
import numpy as np
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def assert_non_singular(self, name='assert_non_singular'):
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    with self._name_scope(name):
        return self._assert_non_singular()