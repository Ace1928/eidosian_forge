import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def order_by(self, attribute):
    """Order the displayed profiler nodes based on a attribute.

    Supported attribute includes micros, bytes, occurrence, params, etc.
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/profiler/g3doc/options.md

    Args:
      attribute: An attribute the profiler node has.
    Returns:
      self
    """
    self._options['order_by'] = attribute
    return self