import collections
from tensorflow.python.util.tf_export import tf_export
@property
def stop_requested(self):
    """Returns whether a stop is requested or not.

    If true, `MonitoredSession` stops iterations.
    Returns:
      A `bool`
    """
    return self._stop_requested