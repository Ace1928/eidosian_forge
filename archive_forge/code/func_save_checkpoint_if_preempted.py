import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def save_checkpoint_if_preempted(self, *args, **kwargs):
    """Saves a checkpoint if a preemption signal has been made available.

    This is an alternative API for `PreemptionCheckpointHandler.run` and
    `PreemptionCheckpointHandler.watch_preemption_scope`. This method works for
    both `tf.distribute.MultiWorkerMirroredStrategy` and
    `tf.distribute.TPUStrategy`. However, **for TPUStrategy, this method will
    add a synchronization point between workers and the coordinator** and thus
    may have performance implication. If this is a concern, use the combination
    of `PreemptionCheckpointHandler.watch_preemption_scope` and
    `PreemptionCheckpointHandler.run` instead.

    ```python
    strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    # initialization omitted

    with strategy.scope():
      # Save in the checkpoint.
      trained_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='trained_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

      checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1)
      preemption_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint_manager)

    while trained_step.numpy() < NUM_STEPS:
      # Train STEPS_IN_FUNCTION steps at once.
      train_multi_step_function()
      trained_step.assign_add(STEPS_IN_FUNCTION)
      preemption_handler.save_checkpoint_if_preempted()
    ```

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
        try:
            with context.async_scope():
                gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
        except errors.AbortedError as abort_error:
            if abort_error.experimental_payloads.get(b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'):
                logging.info('Clearing preemption error to save checkpoint...')
                context.async_clear_error()
                self._save_checkpoint(*args, **kwargs)
                self._exit_fn()
            else:
                raise
    elif self._platform_device in (failure_handling_util.PlatformDevice.GCE_TPU, failure_handling_util.PlatformDevice.GCE_CPU):
        return
    else:
        self._check_preemption_and_maybe_checkpoint(*args, **kwargs)
        self._run_counter += 1
        self._estimated_run_time = 0