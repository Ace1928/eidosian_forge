from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
@abc.abstractmethod
def get_start_state(self):
    """Returns a tuple of state for the start of the time series.

    For example, a mean and covariance. State should not have a batch
    dimension, and will often be TensorFlow Variables to be learned along with
    the rest of the model parameters.
    """
    pass