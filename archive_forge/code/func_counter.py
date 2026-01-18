import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def counter(start=0, step=1, dtype=dtypes.int64, name=None):
    """Creates a `Dataset` that counts from `start` in steps of size `step`.

    Unlike `tf.data.Dataset.range`, which stops at some ending number,
    `tf.data.Dataset.counter` produces elements indefinitely.

    >>> dataset = tf.data.experimental.Counter().take(5)
    >>> list(dataset.as_numpy_iterator())
    [0, 1, 2, 3, 4]
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int64, name=None)
    >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)
    >>> dataset = tf.data.experimental.Counter(start=2).take(5)
    >>> list(dataset.as_numpy_iterator())
    [2, 3, 4, 5, 6]
    >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
    >>> list(dataset.as_numpy_iterator())
    [2, 7, 12, 17, 22]
    >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
    >>> list(dataset.as_numpy_iterator())
    [10, 9, 8, 7, 6]

    Args:
      start: (Optional.) The starting value for the counter. Defaults to 0.
      step: (Optional.) The step size for the counter. Defaults to 1.
      dtype: (Optional.) The data type for counter elements. Defaults to
        `tf.int64`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A `Dataset` of scalar `dtype` elements.
    """
    from tensorflow.python.data.ops import counter_op
    return counter_op._counter(start, step, dtype, name=name)