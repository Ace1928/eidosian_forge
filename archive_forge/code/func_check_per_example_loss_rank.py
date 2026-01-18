from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
@tf_contextlib.contextmanager
def check_per_example_loss_rank(per_example_loss):
    """Context manager that checks that the rank of per_example_loss is at least 1.

  Args:
    per_example_loss: Per example loss tensor.

  Yields:
    A context manager.
  """
    loss_rank = per_example_loss.shape.rank
    if loss_rank is not None:
        if loss_rank == 0:
            raise ValueError(f'Invalid value passed for `per_example_loss`. Expected a tensor with at least rank 1. Received per_example_loss={per_example_loss} with rank {loss_rank}')
        yield
    else:
        with ops.control_dependencies([check_ops.assert_greater_equal(array_ops.rank(per_example_loss), math_ops.cast(1, dtype=dtypes.int32), message='Invalid value passed for `per_example_loss`. Expected a tensor with at least rank 1.')]):
            yield