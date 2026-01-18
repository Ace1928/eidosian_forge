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
def register_session_run_conversion_functions(tensor_type, fetch_function, feed_function=None, feed_function_for_partial_run=None):
    """Register fetch and feed conversion functions for `tf.Session.run()`.

  This function registers a triple of conversion functions for fetching and/or
  feeding values of user-defined types in a call to tf.Session.run().

  An example

  ```python
     class SquaredTensor(object):
       def __init__(self, tensor):
         self.sq = tf.square(tensor)
     #you can define conversion functions as follows:
     fetch_function = lambda squared_tensor:([squared_tensor.sq],
                                             lambda val: val[0])
     feed_function = lambda feed, feed_val: [(feed.sq, feed_val)]
     feed_function_for_partial_run = lambda feed: [feed.sq]
     #then after invoking this register function, you can use as follows:
     session.run(squared_tensor1,
                 feed_dict = {squared_tensor2 : some_numpy_array})
  ```

  Args:
    tensor_type: The type for which you want to register a conversion function.
    fetch_function: A callable that takes an object of type `tensor_type` and
      returns a tuple, where the first element is a list of `tf.Tensor` objects,
      and the second element is a callable that takes a list of ndarrays and
      returns an object of some value type that corresponds to `tensor_type`.
      fetch_function describes how to expand fetch into its component Tensors
      and how to contract the fetched results back into a single return value.
    feed_function: A callable that takes feed_key and feed_value as input, and
      returns a list of tuples (feed_tensor, feed_val), feed_key must have type
      `tensor_type`, and feed_tensor must have type `tf.Tensor`. Each feed
      function describes how to unpack a single fed value and map it to feeds of
      one or more tensors and their corresponding values.
    feed_function_for_partial_run: A callable for specifying tensor values to
      feed when setting up a partial run, which takes a `tensor_type` type
      object as input, and returns a list of Tensors.

  Raises:
    ValueError: If `tensor_type` has already been registered.
  """
    for conversion_function in _REGISTERED_EXPANSIONS:
        if issubclass(conversion_function[0], tensor_type):
            raise ValueError(f'{tensor_type} has already been registered so ignore it.')
    _REGISTERED_EXPANSIONS.insert(0, (tensor_type, fetch_function, feed_function, feed_function_for_partial_run))