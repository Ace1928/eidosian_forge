from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@property
def loss_reduction(self):
    """See `base_head.Head` for details."""
    loss_reductions = [head.loss_reduction for head in self._heads]
    if len(set(loss_reductions)) > 1:
        raise ValueError('The loss_reduction must be the same for different heads. Given: {}'.format(loss_reductions))
    return loss_reductions[0]