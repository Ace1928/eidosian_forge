from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils.sharded_mutable_dense_hashtable import _ShardedMutableDenseHashTable
def regularized_loss(self, examples):
    """Add operations to compute the loss with regularization loss included.

    Args:
      examples: Examples to compute loss on.

    Returns:
      An Operation that computes mean (regularized) loss for given set of
      examples.
    Raises:
      ValueError: if examples are not well defined.
    """
    self._assert_specified(['example_labels', 'example_weights', 'sparse_features', 'dense_features'], examples)
    self._assert_list(['sparse_features', 'dense_features'], examples)
    with name_scope('sdca/regularized_loss'):
        weights = internal_convert_to_tensor(examples['example_weights'])
        return (self._l1_loss() + self._l2_loss()) / tf.math.reduce_sum(tf.cast(weights, tf.dtypes.float64)) + self.unregularized_loss(examples)