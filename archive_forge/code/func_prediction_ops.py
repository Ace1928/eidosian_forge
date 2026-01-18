from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import distributions
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import model
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def prediction_ops(self, times, values, exogenous_regressors):
    """Compute model predictions given input data.

    Args:
      times: A [batch size, self.window_size] integer Tensor, the first
        self.input_window_size times in each part of the batch indicating input
        features, and the last self.output_window_size times indicating
        prediction times.
      values: A [batch size, self.input_window_size, self.num_features] Tensor
        with input features.
      exogenous_regressors: A [batch size, self.window_size,
        self.exogenous_size] Tensor with exogenous features.

    Returns:
      Tuple (predicted_mean, predicted_covariance), where each element is a
      Tensor with shape [batch size, self.output_window_size,
      self.num_features].
    """
    times.get_shape().assert_is_compatible_with([None, self.window_size])
    batch_size = tf.compat.v1.shape(times)[0]
    if self.input_window_size:
        values.get_shape().assert_is_compatible_with([None, self.input_window_size, self.num_features])
    if exogenous_regressors is not None:
        exogenous_regressors.get_shape().assert_is_compatible_with([None, self.window_size, self.exogenous_size])
    input_window_features = []
    input_feature_size = 0
    output_window_features = []
    output_feature_size = 0
    if self._periodicities:
        _, time_features = self._compute_time_features(times)
        num_time_features = self._buckets * len(self._periodicities)
        time_features = tf.reshape(time_features, [batch_size, self.window_size, num_time_features])
        input_time_features, output_time_features = tf.split(time_features, (self.input_window_size, self.output_window_size), axis=1)
        input_feature_size += num_time_features
        output_feature_size += num_time_features
        input_window_features.append(input_time_features)
        output_window_features.append(output_time_features)
    if self.input_window_size:
        inp = tf.slice(values, [0, 0, 0], [-1, self.input_window_size, -1])
        input_window_features.append(tf.reshape(inp, [batch_size, self.input_window_size, self.num_features]))
        input_feature_size += self.num_features
    if self.exogenous_size:
        input_exogenous_features, output_exogenous_features = tf.split(exogenous_regressors, (self.input_window_size, self.output_window_size), axis=1)
        input_feature_size += self.exogenous_size
        output_feature_size += self.exogenous_size
        input_window_features.append(input_exogenous_features)
        output_window_features.append(output_exogenous_features)
    assert input_window_features
    input_window_features = tf.concat(input_window_features, axis=2)
    if output_window_features:
        output_window_features = tf.concat(output_window_features, axis=2)
    else:
        output_window_features = tf.zeros([batch_size, self.output_window_size, 0], dtype=self.dtype)
    static_batch_size = times.get_shape().dims[0].value
    input_window_features.set_shape([static_batch_size, self.input_window_size, input_feature_size])
    output_window_features.set_shape([static_batch_size, self.output_window_size, output_feature_size])
    return self._output_window_predictions(input_window_features, output_window_features)