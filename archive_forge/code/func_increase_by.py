import collections
import functools
import time
from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def increase_by(self, value):
    """Atomically increments the value.

    Args:
      value: non-negative value.
    """
    pywrap_tfe.TFE_MonitoringCounterCellIncrementBy(self._cell, value)