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
def approximate_duality_gap(self):
    """Add operations to compute the approximate duality gap.

    Returns:
      An Operation that computes the approximate duality gap over all
      examples.
    """
    with name_scope('sdca/approximate_duality_gap'):
        _, values_list = self._hashtable.export_sharded()
        shard_sums = []
        for values in values_list:
            with tf.compat.v1.device(values.device):
                with tf.control_dependencies(shard_sums):
                    shard_sums.append(tf.math.reduce_sum(tf.cast(values, dtype=tf.dtypes.float64), 0))
        summed_values = tf.math.add_n(shard_sums)
        primal_loss = summed_values[1]
        dual_loss = summed_values[2]
        example_weights = summed_values[3]
        return (primal_loss + dual_loss + self._l1_loss() + 2.0 * self._l2_loss()) / example_weights