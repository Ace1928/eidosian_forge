from typing import List
import numpy as np
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.types.core import Tensor, TensorLike  # pylint: disable=g-multiple-import
@polymorphic_function.function
def stateless_random_uniform(shape, seed, layout):
    """Creates uniform random tensor with the given layout."""
    return api.relayout(stateless_random_ops.stateless_random_uniform(shape=shape, seed=seed), layout=layout)