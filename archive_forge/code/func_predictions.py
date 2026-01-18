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
def predictions(self, examples):
    """Add operations to compute predictions by the model.

    If logistic_loss is being used, predicted probabilities are returned.
    If poisson_loss is being used, predictions are exponentiated.
    Otherwise, (raw) linear predictions (w*x) are returned.

    Args:
      examples: Examples to compute predictions on.

    Returns:
      An Operation that computes the predictions for examples.

    Raises:
      ValueError: if examples are not well defined.
    """
    self._assert_specified(['example_weights', 'sparse_features', 'dense_features'], examples)
    self._assert_list(['sparse_features', 'dense_features'], examples)
    result = self._linear_predictions(examples)
    if self._options['loss_type'] == 'logistic_loss':
        with name_scope('sdca/logistic_prediction'):
            result = tf.math.sigmoid(result)
    elif self._options['loss_type'] == 'poisson_loss':
        with name_scope('sdca/poisson_prediction'):
            result = tf.math.exp(result)
    return result