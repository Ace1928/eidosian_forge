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
@tf_export(v1=['nn.sufficient_statistics'])
@dispatch.add_dispatch_support
def sufficient_statistics(x, axes, shift=None, keep_dims=None, name=None, keepdims=None):
    """Calculate the sufficient statistics for the mean and variance of `x`.

  These sufficient statistics are computed using the one pass algorithm on
  an input that's optionally shifted. See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

  For example:
  >>> t = [[1, 2, 3], [4, 5, 6]]
  >>> sufficient_statistics(t, [1])
  (<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(2,),
  dtype=int32, numpy=array([ 6, 15], dtype=int32)>, <tf.Tensor: shape=(2,),
  dtype=int32, numpy=array([14, 77], dtype=int32)>, None)
  >>> sufficient_statistics(t, [-1])
  (<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(2,),
  dtype=int32, numpy=array([ 6, 15], dtype=int32)>, <tf.Tensor: shape=(2,),
  dtype=int32, numpy=array([14, 77], dtype=int32)>, None)

  Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance. As in
      Python, the axes can also be negative numbers. A negative axis is
      interpreted as counting from the end of the rank, i.e., axis +
      rank(values)-th dimension.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keep_dims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.
    keepdims: Alias for keep_dims.

  Returns:
    Four `Tensor` objects of the same type as `x`:

    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
  """
    axes = list(set(axes))
    keep_dims = deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    if keep_dims is None:
        keep_dims = False
    with ops.name_scope(name, 'sufficient_statistics', [x, shift]):
        x = ops.convert_to_tensor(x, name='x')
        x_shape = x.get_shape()
        if x_shape.rank is not None and all((x_shape.dims[d].value is not None for d in axes)):
            counts = 1
            for d in axes:
                counts *= x_shape.dims[d].value
            counts = constant_op.constant(counts, dtype=x.dtype)
        else:
            rank = array_ops.rank(x)
            positive_axes = [axis + rank if axis < 0 else axis for axis in axes]
            x_dims = array_ops.gather(math_ops.cast(array_ops.shape(x), x.dtype), positive_axes)
            counts = math_ops.reduce_prod(x_dims, name='count')
        if shift is not None:
            shift = ops.convert_to_tensor(shift, name='shift')
            m_ss = math_ops.subtract(x, shift)
            v_ss = math_ops.squared_difference(x, shift)
        else:
            m_ss = x
            v_ss = math_ops.square(x)
        m_ss = math_ops.reduce_sum(m_ss, axes, keepdims=keep_dims, name='mean_ss')
        v_ss = math_ops.reduce_sum(v_ss, axes, keepdims=keep_dims, name='var_ss')
    return (counts, m_ss, v_ss, shift)