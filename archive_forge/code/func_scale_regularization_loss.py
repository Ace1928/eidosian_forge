import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.scale_regularization_loss')
@dispatch.add_dispatch_support
def scale_regularization_loss(regularization_loss):
    """Scales the sum of the given regularization losses by number of replicas.

  Usage with distribution strategy and custom training loop:

  ```python
  with strategy.scope():
    def compute_loss(self, label, predictions):
      per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)

      # Compute loss that is scaled by sample_weight and by global batch size.
      loss = tf.nn.compute_average_loss(
          per_example_loss,
          sample_weight=sample_weight,
          global_batch_size=GLOBAL_BATCH_SIZE)

      # Add scaled regularization losses.
      loss += tf.nn.scale_regularization_loss(tf.nn.l2_loss(weights))
      return loss
  ```

  Args:
    regularization_loss: Regularization loss.

  Returns:
    Scalar loss value.
  """
    if distribute_lib.has_strategy() and distribute_lib.in_cross_replica_context():
        raise RuntimeError('You are calling `scale_regularization_loss` in cross replica context, while it was expected to be called in replica context.')
    num_replicas = distribute_lib.get_strategy().num_replicas_in_sync
    return math_ops.reduce_sum(regularization_loss) / num_replicas