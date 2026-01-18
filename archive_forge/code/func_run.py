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
def run(self):
    """Executes the run_foo for task type `foo`.

    `_TrainingExecutor` predefines the procedure for task type 'chief',
    'worker', 'ps', and 'evaluator'. For task type `foo`, the corresponding
    procedure is `run_foo'. This `run` method invoke the procedure base on the
    `RunConfig.task_type`.

    Returns:
      A tuple of the result of the `evaluate` call to the `Estimator` and the
      export results using the specified `ExportStrategy`.
      Currently undefined for distributed training mode.

    Raises:
      ValueError: if the estimator.config is mis-configured.
    """
    config = self._estimator.config
    if not config.cluster_spec and config.task_type != run_config_lib.TaskType.EVALUATOR:
        tf.compat.v1.logging.info('Running training and evaluation locally (non-distributed).')
        return self.run_local()
    if not config.task_type:
        raise ValueError('`estimator.config` must have task_type set. This usually means TF_CONFIG environment is not set correctly.')
    if config.task_type == 'local':
        raise ValueError('`task.type` in TF_CONFIG cannot be `local`. Leaving `cluster` and `task` properties in TF_CONFIG absent triggers train and evaluate `Estimator` locally (non-distributed).')
    available_tasks = [x for x in dir(self) if x.startswith('run_') and x != 'run_local' and callable(getattr(self, x))]
    task_to_run = 'run_' + config.task_type
    if task_to_run not in available_tasks:
        raise ValueError('Task type {} is not supported. Supported task types are {}'.format(config.task_type, [x[len('run_'):] for x in available_tasks]))
    getattr(self, task_to_run)()