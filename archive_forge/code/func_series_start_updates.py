from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def series_start_updates():
    mean, variance = tf.compat.v1.nn.moments(values[min_time_batch, :self._starting_variance_window_size], axes=[0])
    return tf.group(tf.compat.v1.assign(statistics.series_start_moments.mean, mean), tf.compat.v1.assign(statistics.series_start_moments.variance, variance))