import math
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import util as losses_util
from tensorflow.python.platform import device_context
from tensorflow.python.util import dispatch
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
@tf_export('nn.normalize_moments')
@dispatch.add_dispatch_support
def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
    """Calculate the mean and variance of based on the sufficient statistics.

  Args:
    counts: A `Tensor` containing the total count of the data (one value).
    mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
      shifted) sum of the elements to average over.
    variance_ss: A `Tensor` containing the variance sufficient statistics: the
      (possibly shifted) squared sum of the data to compute the variance over.
    shift: A `Tensor` containing the value by which the data is shifted for
      numerical stability, or `None` if no shift was performed.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
    with ops.name_scope(name, 'normalize', [counts, mean_ss, variance_ss, shift]):
        divisor = math_ops.reciprocal(counts, name='divisor')
        if shift is not None:
            shifted_mean = math_ops.multiply(mean_ss, divisor, name='shifted_mean')
            mean = math_ops.add(shifted_mean, shift, name='mean')
        else:
            shifted_mean = math_ops.multiply(mean_ss, divisor, name='mean')
            mean = shifted_mean
        variance = math_ops.subtract(math_ops.multiply(variance_ss, divisor), math_ops.square(shifted_mean), name='variance')
    return (mean, variance)