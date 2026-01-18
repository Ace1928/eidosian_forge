from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.limit_epochs'])
@deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.')
def limit_epochs(tensor, num_epochs=None, name=None):
    """Returns tensor `num_epochs` times and then raises an `OutOfRange` error.

  Note: creates local counter `epochs`. Use `local_variables_initializer()` to
  initialize local variables.

  Args:
    tensor: Any `Tensor`.
    num_epochs: A positive integer (optional).  If specified, limits the number
      of steps the output tensor may be evaluated.
    name: A name for the operations (optional).

  Returns:
    tensor or `OutOfRange`.

  Raises:
    ValueError: if `num_epochs` is invalid.
  """
    if num_epochs is None:
        return tensor
    if num_epochs <= 0:
        raise ValueError('num_epochs must be > 0 not %d.' % num_epochs)
    with ops.name_scope(name, 'limit_epochs', [tensor]) as name:
        zero64 = constant_op.constant(0, dtype=dtypes.int64)
        epochs = variable_v1.VariableV1(zero64, name='epochs', trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES])
        counter = epochs.count_up_to(num_epochs)
        with ops.control_dependencies([counter]):
            return array_ops.identity(tensor, name=name)