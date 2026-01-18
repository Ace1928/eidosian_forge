import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def ragged_factory(values, row_splits):
    row_splits = constant_op.constant(row_splits, dtype=row_splits_dtype)
    return ragged_tensor.RaggedTensor.from_row_splits(values, row_splits, validate=False)