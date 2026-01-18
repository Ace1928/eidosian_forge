import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.pinv')
@dispatch.add_dispatch_support
def pinv(a, rcond=None, validate_args=False, name=None):
    """Compute the Moore-Penrose pseudo-inverse of one or more matrices.

  Calculate the [generalized inverse of a matrix](
  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) using its
  singular-value decomposition (SVD) and including all large singular values.

  The pseudo-inverse of a matrix `A`, is defined as: 'the matrix that 'solves'
  [the least-squares problem] `A @ x = b`,' i.e., if `x_hat` is a solution, then
  `A_pinv` is the matrix such that `x_hat = A_pinv @ b`. It can be shown that if
  `U @ Sigma @ V.T = A` is the singular value decomposition of `A`, then
  `A_pinv = V @ inv(Sigma) U^T`. [(Strang, 1980)][1]

  This function is analogous to [`numpy.linalg.pinv`](
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html).
  It differs only in default value of `rcond`. In `numpy.linalg.pinv`, the
  default `rcond` is `1e-15`. Here the default is
  `10. * max(num_rows, num_cols) * np.finfo(dtype).eps`.

  Args:
    a: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
      pseudo-inverted.
    rcond: `Tensor` of small singular value cutoffs.  Singular values smaller
      (in modulus) than `rcond` * largest_singular_value (again, in modulus) are
      set to zero. Must broadcast against `tf.shape(a)[:-2]`.
      Default value: `10. * max(num_rows, num_cols) * np.finfo(a.dtype).eps`.
    validate_args: When `True`, additional assertions might be embedded in the
      graph.
      Default value: `False` (i.e., no graph assertions are added).
    name: Python `str` prefixed to ops created by this function.
      Default value: 'pinv'.

  Returns:
    a_pinv: (Batch of) pseudo-inverse of input `a`. Has same shape as `a` except
      rightmost two dimensions are transposed.

  Raises:
    TypeError: if input `a` does not have `float`-like `dtype`.
    ValueError: if input `a` has fewer than 2 dimensions.

  #### Examples

  ```python
  import tensorflow as tf
  import tensorflow_probability as tfp

  a = tf.constant([[1.,  0.4,  0.5],
                   [0.4, 0.2,  0.25],
                   [0.5, 0.25, 0.35]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]], dtype=float32)

  a = tf.constant([[1.,  0.4,  0.5,  1.],
                   [0.4, 0.2,  0.25, 2.],
                   [0.5, 0.25, 0.35, 3.]])
  tf.matmul(tf.linalg.pinv(a), a)
  # ==> array([[ 0.76,  0.37,  0.21, -0.02],
               [ 0.37,  0.43, -0.33,  0.02],
               [ 0.21, -0.33,  0.81,  0.01],
               [-0.02,  0.02,  0.01,  1.  ]], dtype=float32)
  ```

  #### References

  [1]: G. Strang. 'Linear Algebra and Its Applications, 2nd Ed.' Academic Press,
       Inc., 1980, pp. 139-142.
  """
    with ops.name_scope(name or 'pinv'):
        a = ops.convert_to_tensor(a, name='a')
        assertions = _maybe_validate_matrix(a, validate_args)
        if assertions:
            with ops.control_dependencies(assertions):
                a = array_ops.identity(a)
        dtype = a.dtype.as_numpy_dtype
        if rcond is None:

            def get_dim_size(dim):
                dim_val = tensor_shape.dimension_value(a.shape[dim])
                if dim_val is not None:
                    return dim_val
                return array_ops.shape(a)[dim]
            num_rows = get_dim_size(-2)
            num_cols = get_dim_size(-1)
            if isinstance(num_rows, int) and isinstance(num_cols, int):
                max_rows_cols = float(max(num_rows, num_cols))
            else:
                max_rows_cols = math_ops.cast(math_ops.maximum(num_rows, num_cols), dtype)
            rcond = 10.0 * max_rows_cols * np.finfo(dtype).eps
        rcond = ops.convert_to_tensor(rcond, dtype=dtype, name='rcond')
        [singular_values, left_singular_vectors, right_singular_vectors] = svd(a, full_matrices=False, compute_uv=True)
        cutoff = rcond * math_ops.reduce_max(singular_values, axis=-1)
        singular_values = array_ops.where_v2(singular_values > array_ops.expand_dims_v2(cutoff, -1), singular_values, np.array(np.inf, dtype))
        a_pinv = math_ops.matmul(right_singular_vectors / array_ops.expand_dims_v2(singular_values, -2), left_singular_vectors, adjoint_b=True)
        if a.shape is not None and a.shape.rank is not None:
            a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))
        return a_pinv