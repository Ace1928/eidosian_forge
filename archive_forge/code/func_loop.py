import contextlib
import sys
import threading
import time
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def loop(coord, timer_interval_secs, target, args=None, kwargs=None):
    """Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(args)`
    repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
    seconds.  The thread terminates when a stop of the coordinator is
    requested.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    """
    looper = LooperThread(coord, timer_interval_secs, target=target, args=args, kwargs=kwargs)
    looper.start()
    return looper