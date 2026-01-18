import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def build_results(self, session, tensor_values):
    """Build results matching the original fetch shape.

    `tensor_values` must be a list of the same length as
    the one returned by `fetches()`, and holding the requested
    fetch values.

    This method builds a struct with the same shape as the original `fetches`
    passed to the constructor, in which the fetches are replaced by their
    fetched value.

    Args:
      session: The enclosing session.  Used for tensor handles.
      tensor_values: List of values matching the list returned by fetches().

    Returns:
      A structure of the same shape as the original `fetches` argument but
        containing tensors or None (for fetched ops).
    """
    full_values = []
    assert len(self._final_fetches) == len(tensor_values)
    i = 0
    j = 0
    for is_op in self._ops:
        if is_op:
            full_values.append(None)
        else:
            if self._fetches[i].ref() in self._feed_handles:
                value = self._feed_handles[self._fetches[i].ref()].eval()
            else:
                value = self._feeds.get(self._fetches[i].ref())
            if value is None:
                value = tensor_values[j]
                j += 1
            dtype = self._fetch_handles.get(self._fetches[i].ref())
            if dtype:
                full_values.append(session_ops.TensorHandle(value, dtype, session))
            else:
                full_values.append(value)
            i += 1
    assert j == len(tensor_values)
    return self._fetch_mapper.build_results(full_values)