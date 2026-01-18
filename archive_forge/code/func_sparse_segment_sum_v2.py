import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('sparse.segment_sum', v1=[])
def sparse_segment_sum_v2(data, indices, segment_ids, num_segments=None, name=None):
    """Computes the sum along sparse segments of a tensor.

  Read [the section on
  segmentation](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/math#about_segmentation)
  for an explanation of segments.

  Like `tf.math.segment_sum`, but `segment_ids` can have rank less than `data`'s
  first dimension, selecting a subset of dimension 0, specified by `indices`.
  `segment_ids` is allowed to have missing ids, in which case the output will
  be zeros at those indices. In those cases `num_segments` is used to determine
  the size of the output.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]

  # With missing segment ids.
  tf.sparse.segment_sum(c, tf.constant([0, 1]), tf.constant([0, 2]),
                        num_segments=4)
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]

  # Select all rows, two segments.
  tf.sparse.segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]

  # Which is equivalent to:
  tf.math.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor` with data that will be assembled in the output.
    indices: A 1-D `Tensor` with indices into `data`. Has same rank as
      `segment_ids`.
    segment_ids: A 1-D `Tensor` with indices into the output `Tensor`. Values
      should be sorted and can be repeated.
    num_segments: An optional int32 scalar. Indicates the size of the output
      `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `tensor` of the shape as data, except for dimension 0 which
    has size `k`, the number of segments specified via `num_segments` or
    inferred for the last element in `segments_ids`.
  """
    return sparse_segment_sum(data, indices, segment_ids, name=name, num_segments=num_segments)