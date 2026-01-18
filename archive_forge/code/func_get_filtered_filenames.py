import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
def get_filtered_filenames(self):
    if self._cached_set is not None:
        return self._cached_set
    filtered_filenames = frozenset((self._filename,))
    if self.parent is not None:
        filtered_filenames |= self.parent.get_filtered_filenames()
    self._cached_set = filtered_filenames
    return filtered_filenames