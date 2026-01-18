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
@tf_export('nn.moments', v1=[])
@dispatch.add_dispatch_support
def moments_v2(x, axes, shift=None, keepdims=False, name=None):
    """Calculates the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  Note: shift is currently not used; the true mean is computed and used.

  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):

   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).

  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: Not used in the current implementation.
    keepdims: produce moments with the same dimensionality as the input.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
    return moments(x=x, axes=axes, shift=shift, name=name, keep_dims=keepdims)