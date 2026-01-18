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
def assert_positive_definite(self, name='assert_positive_definite'):
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    with self._name_scope(name):
        return self._assert_positive_definite()