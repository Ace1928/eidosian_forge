from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.experimental.make_stop_at_checkpoint_step_hook')
def make_stop_at_checkpoint_step_hook(estimator, last_step, wait_after_file_check_secs=30):
    """Creates a proper StopAtCheckpointStepHook based on chief status."""
    if estimator.config.is_chief:
        return tf.compat.v1.train.StopAtStepHook(last_step=last_step)
    return _StopAtCheckpointStepHook(model_dir=estimator.model_dir, last_step=last_step, wait_after_file_check_secs=wait_after_file_check_secs)