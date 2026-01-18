from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def as_signature_def(self, receiver_tensors):
    no_state_receiver_tensors = {key: value for key, value in receiver_tensors.items() if not key.startswith(feature_keys.State.STATE_PREFIX)}
    return super(_NoStatePredictOutput, self).as_signature_def(receiver_tensors=no_state_receiver_tensors)