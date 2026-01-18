import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(math_ops.tensor_not_equals)
def tensor_not_equals(self: ragged_tensor.RaggedOrDense, other: ragged_tensor.RaggedOrDense):
    """Ragged version of the operation invoked by `Tensor.__ne__`."""
    if other is None:
        return False
    elif _use_legacy_mode_for_tensor_equality(self):
        return self is not other
    else:
        try:
            return math_ops.not_equal(self, other)
        except (errors.InvalidArgumentError, ValueError):
            return True