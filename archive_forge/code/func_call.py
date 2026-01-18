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
def call(self, input_window_features, output_window_features):
    """Compute predictions from input and output windows."""
    _, state_h, state_c = self._encoder(input_window_features)
    encoder_states = [state_h, state_c]
    decoder_output = self._decoder(output_window_features, initial_state=encoder_states)
    predicted_mean = self._mean_transform(decoder_output)
    predicted_covariance = gen_math_ops.exp(self._covariance_transform(decoder_output))
    return {'mean': predicted_mean, 'covariance': predicted_covariance}