from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import sys
import threading
import time
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.tools import analytics
def record_done(self, source):
    """Mark execution source `source` as done.

    If an error was originally reported from `source` it is left intact.

    Args:
      source: `str`, source being recorded
    """
    tf.compat.v1.logging.info('%s marked as finished', source)
    if source not in self._errors:
        self._errors[source] = None