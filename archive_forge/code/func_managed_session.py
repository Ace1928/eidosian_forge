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
@contextlib.contextmanager
def managed_session(self, master='', config=None, start_standard_services=True, close_summary_writer=True):
    """Returns a context manager for a managed session.

    This context manager creates and automatically recovers a session.  It
    optionally starts the standard services that handle checkpoints and
    summaries.  It monitors exceptions raised from the `with` block or from the
    services and stops the supervisor as needed.

    The context manager is typically used as follows:

    ```python
    def train():
      sv = tf.compat.v1.train.Supervisor(...)
      with sv.managed_session(<master>) as sess:
        for step in range(..):
          if sv.should_stop():
            break
          sess.run(<my training op>)
          ...do other things needed at each training step...
    ```

    An exception raised from the `with` block or one of the service threads is
    raised again when the block exits.  This is done after stopping all threads
    and closing the session.  For example, an `AbortedError` exception, raised
    in case of preemption of one of the workers in a distributed model, is
    raised again when the block exits.

    If you want to retry the training loop in case of preemption you can do it
    as follows:

    ```python
    def main(...):
      while True
        try:
          train()
        except tf.errors.Aborted:
          pass
    ```

    As a special case, exceptions used for control flow, such as
    `OutOfRangeError` which reports that input queues are exhausted, are not
    raised again from the `with` block: they indicate a clean termination of
    the training loop and are considered normal termination.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional `ConfigProto` proto used to configure the session. Passed
        as-is to create the session.
      start_standard_services: Whether to start the standard services, such as
        checkpoint, summary and step counter.
      close_summary_writer: Whether to close the summary writer when closing the
        session.  Defaults to True.

    Returns:
      A context manager that yields a `Session` restored from the latest
      checkpoint or initialized from scratch if not checkpoint exists.  The
      session is closed when the `with` block exits.
    """
    try:
        sess = self.prepare_or_wait_for_session(master=master, config=config, start_standard_services=start_standard_services)
        yield sess
    except Exception as e:
        self.request_stop(e)
    finally:
        try:
            self.stop(close_summary_writer=close_summary_writer)
        finally:
            try:
                sess.close()
            except Exception:
                pass