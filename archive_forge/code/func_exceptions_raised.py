import threading
import weakref
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@property
def exceptions_raised(self):
    """Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised