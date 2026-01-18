import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def lu_reconstruct_assertions(lower_upper, perm, validate_args):
    """Returns list of assertions related to `lu_reconstruct` assumptions."""
    assertions = []
    message = 'Input `lower_upper` must have at least 2 dimensions.'
    if lower_upper.shape.rank is not None and lower_upper.shape.rank < 2:
        raise ValueError(message)
    elif validate_args:
        assertions.append(check_ops.assert_rank_at_least_v2(lower_upper, rank=2, message=message))
    message = '`rank(lower_upper)` must equal `rank(perm) + 1`'
    if lower_upper.shape.rank is not None and perm.shape.rank is not None:
        if lower_upper.shape.rank != perm.shape.rank + 1:
            raise ValueError(message)
    elif validate_args:
        assertions.append(check_ops.assert_rank(lower_upper, rank=array_ops.rank(perm) + 1, message=message))
    message = '`lower_upper` must be square.'
    if lower_upper.shape[:-2].is_fully_defined():
        if lower_upper.shape[-2] != lower_upper.shape[-1]:
            raise ValueError(message)
    elif validate_args:
        m, n = array_ops.split(array_ops.shape(lower_upper)[-2:], num_or_size_splits=2)
        assertions.append(check_ops.assert_equal(m, n, message=message))
    return assertions