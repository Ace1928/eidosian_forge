from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def end(self, session):
    """Runs evaluator for final model."""
    self._evaluate(session)