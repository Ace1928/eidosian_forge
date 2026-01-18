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
def start_standard_services(self, sess):
    """Start the standard services for 'sess'.

    This starts services in the background.  The services started depend
    on the parameters to the constructor and may include:

      - A Summary thread computing summaries every save_summaries_secs.
      - A Checkpoint thread saving the model every save_model_secs.
      - A StepCounter thread measure step time.

    Args:
      sess: A Session.

    Returns:
      A list of threads that are running the standard services.  You can use
      the Supervisor's Coordinator to join these threads with:
        sv.coord.Join(<list of threads>)

    Raises:
      RuntimeError: If called with a non-chief Supervisor.
      ValueError: If not `logdir` was passed to the constructor as the
        services need a log directory.
    """
    if not self._is_chief:
        raise RuntimeError('Only chief supervisor can start standard services. Because only chief supervisors can write events.')
    if not self._logdir:
        logging.warning("Standard services need a 'logdir' passed to the SessionManager")
        return
    if self._global_step is not None and self._summary_writer:
        current_step = training_util.global_step(sess, self._global_step)
        self._summary_writer.add_session_log(SessionLog(status=SessionLog.START), current_step)
    threads = []
    if self._save_summaries_secs and self._summary_writer:
        if self._summary_op is not None:
            threads.append(SVSummaryThread(self, sess))
        if self._global_step is not None:
            threads.append(SVStepCounterThread(self, sess))
    if self.saver and self._save_model_secs:
        threads.append(SVTimerCheckpointThread(self, sess))
    for t in threads:
        t.start()
    return threads