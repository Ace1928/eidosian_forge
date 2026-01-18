from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
@property
def master_job(self):
    """Returns the job name to use to place TPU computations on.

    Returns:
      A string containing the job name, or None if no job should be specified.

    Raises:
      ValueError: If the user needs to specify a tpu_job_name, because we are
        unable to infer the job name automatically, or if the user-specified job
        names are inappropriate.
    """
    run_config = self._config
    if run_config.tpu_config.tpu_job_name:
        return run_config.tpu_config.tpu_job_name
    mode = self._assert_mode()
    master = run_config.evaluation_master if mode == model_fn_lib.ModeKeys.EVAL else run_config.master
    cluster_def = run_config.session_config.cluster_def if run_config.session_config else None
    try:
        master_job = tpu_system_metadata_lib.master_job(master, cluster_def)
    except ValueError as e:
        raise ValueError(str(e) + ' Please specify a tpu_job_name as part of your TPUConfig.')
    return master_job