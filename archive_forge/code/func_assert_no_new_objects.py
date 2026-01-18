import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
@trace.trace_wrapper
def assert_no_new_objects(self, threshold=None):
    """Assert no new Python objects."""
    if not threshold:
        threshold = {}
    count_diff = self._snapshot_diff(0, -1)
    original_count_diff = copy.deepcopy(count_diff)
    count_diff.subtract(collections.Counter(threshold))
    if max(count_diff.values() or [0]) > 0:
        raise AssertionError(f'New Python objects created exceeded the threshold.\nPython object threshold:\n{threshold}\n\nNew Python objects:\n{original_count_diff.most_common()}')
    elif min(count_diff.values(), default=0) < 0:
        logging.warning(f'New Python objects created were less than the threshold.\nPython object threshold:\n{threshold}\n\nNew Python objects:\n{original_count_diff.most_common()}')