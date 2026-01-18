from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def initialize_graph(self, input_statistics=None):
    """Define ops for the model, not depending on any previously defined ops.

    Args:
      input_statistics: A math_utils.InputStatistics object containing input
        statistics. If None, data-independent defaults are used, which may
        result in longer or unstable training.
    """
    self._graph_initialized = True
    self._input_statistics = input_statistics
    if self._input_statistics:
        self._stats_means, variances = self._input_statistics.overall_feature_moments
        self._stats_sigmas = tf.math.sqrt(variances)