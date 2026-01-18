import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def stop_on_exception(self):
    """Context manager to request stop when an Exception is raised.

    Code that uses a coordinator must catch exceptions and pass
    them to the `request_stop()` method to stop the other threads
    managed by the coordinator.

    This context handler simplifies the exception handling.
    Use it as follows:

    ```python
    with coord.stop_on_exception():
      # Any exception raised in the body of the with
      # clause is reported to the coordinator before terminating
      # the execution of the body.
      ...body...
    ```

    This is completely equivalent to the slightly longer code:

    ```python
    try:
      ...body...
    except:
      coord.request_stop(sys.exc_info())
    ```

    Yields:
      nothing.
    """
    try:
        yield
    except:
        self.request_stop(ex=sys.exc_info())