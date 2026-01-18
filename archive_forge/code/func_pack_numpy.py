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
def pack_numpy(value: np.ndarray, layout: layout_lib.Layout, make_sparse: bool=False) -> Tensor:
    assert value is not None
    unpacked = unpack(value, layout)
    if make_sparse:
        return api.pack([sparse_ops.from_dense(t) for t in unpacked], layout)
    return api.pack(unpacked, layout)