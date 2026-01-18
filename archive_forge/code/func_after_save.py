from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def after_save(self, session, global_step_value):
    del session
    if self._is_first_run:
        self._is_first_run = False
        return
    if not self._continuous_eval_listener.before_eval():
        tf.compat.v1.logging.info('Exiting training and evaluation loop, as requested by _ContinuousEvalListener.before_eval.')
        return True
    if self._timer.should_trigger_for_step(global_step_value):
        self._evaluate(global_step_value)
        if not self._continuous_eval_listener.after_eval(self.eval_result):
            tf.compat.v1.logging.info('Exiting evaluation, as requested by _ContinuousEvalListener.after_eval.')
            return True
    else:
        tf.compat.v1.logging.info('Skip the current checkpoint eval due to throttle secs ({} secs).'.format(self._eval_throttle_secs))