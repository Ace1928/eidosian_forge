import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def embed_check_nonnegative_integer_form(x, name='embed_check_nonnegative_integer_form'):
    """Assert x is a non-negative tensor, and optionally of integers."""
    with ops.name_scope(name, values=[x]):
        x = ops.convert_to_tensor(x, name='x')
        assertions = [check_ops.assert_non_negative(x, message="'{}' must be non-negative.".format(x))]
        if not x.dtype.is_integer:
            assertions += [assert_integer_form(x, message="'{}' cannot contain fractional components.".format(x))]
        return control_flow_ops.with_dependencies(assertions, x)