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
@deprecated('2016-11-30', 'Please switch to tf.summary.merge_all.')
def merge_all_summaries(key=ops.GraphKeys.SUMMARIES):
    """Merges all summaries collected in the default graph.

  This op is deprecated. Please switch to tf.compat.v1.summary.merge_all, which
  has
  identical behavior.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
    summary_ops = ops.get_collection(key)
    if not summary_ops:
        return None
    else:
        return merge_summary(summary_ops)