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
def run_master(self):
    """Runs task master."""
    _assert_eval_spec(self._eval_spec)
    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec, self._train_spec.max_steps)
    saving_listeners = self._train_spec.saving_listeners + tuple([_NewCheckpointListenerForEvaluate(evaluator, self._eval_spec.throttle_secs, _ContinuousEvalListener())])
    self._start_distributed_training(saving_listeners=saving_listeners)