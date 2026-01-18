from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
@property
def exogenous_feature_columns(self):
    """`tf.feature_colum`s for features which are not predicted."""
    return self._exogenous_feature_columns