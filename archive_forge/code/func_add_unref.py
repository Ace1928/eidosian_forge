import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def add_unref(self, timestamp):
    """Adds an unref to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object unreference as an integer.
    """
    self._unref_times.append(timestamp)