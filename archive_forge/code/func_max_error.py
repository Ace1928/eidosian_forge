import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def max_error(grad1, grad2):
    """Computes maximum elementwise gap.

  Computes the maximum elementwise gap between two lists of tensors of the same
  shape.

  Args:
    grad1: a lists of tensors.
    grad2: a lists of tensors with the same shape as grad1.

  Returns:
    The maximum elementwise gap between the two.
  """
    error = 0
    for j_t, j_n in zip(grad1, grad2):
        if j_t.size or j_n.size:
            error = np.maximum(error, np.fabs(j_t - j_n).max())
    return error