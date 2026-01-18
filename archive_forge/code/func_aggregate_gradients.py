import abc
import platform
import re
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src import initializers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def aggregate_gradients(self, grads_and_vars):
    """Aggregate gradients on all devices.

        By default, we will perform reduce_sum of gradients across devices.
        Users can implement their own aggregation logic by overriding this
        method.

        Args:
          grads_and_vars: List of (gradient, variable) pairs.

        Returns:
          List of (gradient, variable) pairs.
        """
    if self._mesh or self._run_with_dtensor:
        logging.warning('Calling aggregate_gradients is unnecessary when the model is used with DTensor, which includes aggregation of replicated gradients as part of backward pass.')
        return grads_and_vars
    else:
        return optimizer_utils.all_reduce_sum_gradients(grads_and_vars)