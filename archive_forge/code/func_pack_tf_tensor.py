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
def pack_tf_tensor(value: Tensor, layout: layout_lib.Layout) -> Tensor:
    if value is None:
        raise ValueError('pack requires values to be passed in')
    unpacked = unpack(value, layout, split_fn=array_ops.split, stack_fn=array_ops_stack.stack)
    return api.pack(unpacked, layout)