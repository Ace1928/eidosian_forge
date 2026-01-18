import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def start_queue_runners(self, sess, queue_runners=None):
    """Start threads for `QueueRunners`.

    Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
    are already started automatically when you create a session with the
    supervisor, so unless you have non-collected queue runners to start
    you do not need to call this explicitly.

    Args:
      sess: A `Session`.
      queue_runners: A list of `QueueRunners`. If not specified, we'll use the
        list of queue runners gathered in the graph under the key
        `GraphKeys.QUEUE_RUNNERS`.

    Returns:
      The list of threads started for the `QueueRunners`.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
    Queues are not compatible with eager execution. To ingest data when eager
    execution is enabled, use the `tf.data` API.
    @end_compatibility
    """
    if context.executing_eagerly():
        raise RuntimeError('Queues are not compatible with eager execution.')
    if queue_runners is None:
        queue_runners = self._graph.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
    threads = []
    for qr in queue_runners:
        threads.extend(qr.create_threads(sess, coord=self._coord, daemon=True, start=True))
    return threads