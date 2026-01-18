import collections as py_collections
import os
import pprint
import random
import sys
from absl import logging
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_logging_ops import *
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
@deprecated('2016-11-30', 'Please switch to tf.summary.histogram. Note that tf.summary.histogram uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.')
def histogram_summary(tag, values, collections=None, name=None):
    """Outputs a `Summary` protocol buffer with a histogram.

  This ops is deprecated. Please switch to tf.summary.histogram.

  For an explanation of why this op was deprecated, and information on how to
  migrate, look
  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    tag: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
    values: A real numeric `Tensor`. Any shape. Values to use to build the
      histogram.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
    with ops.name_scope(name, 'HistogramSummary', [tag, values]) as scope:
        val = gen_logging_ops.histogram_summary(tag=tag, values=values, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val