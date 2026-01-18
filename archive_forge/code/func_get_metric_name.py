import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def get_metric_name(metric, weighted=False):
    """Returns the name corresponding to the given metric input.

  Args:
    metric: Metric function name or reference.
    weighted: Boolean indicating if the given metric is weighted.

  Returns:
      The metric name.
  """
    if tf2.enabled():
        if isinstance(metric, str):
            return metric
        metric = metrics_module.get(metric)
        return metric.name if hasattr(metric, 'name') else metric.__name__
    else:
        metric_name_prefix = 'weighted_' if weighted else ''
        if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
            if metric in ('accuracy', 'acc'):
                suffix = 'acc'
            elif metric in ('crossentropy', 'ce'):
                suffix = 'ce'
        else:
            metric_fn = metrics_module.get(metric)
            if hasattr(metric_fn, 'name'):
                suffix = metric_fn.name
            else:
                suffix = metric_fn.__name__
        metric_name = metric_name_prefix + suffix
        return metric_name