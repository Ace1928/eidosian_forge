import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_linalg_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.cholesky_solve', v1=['linalg.cholesky_solve', 'cholesky_solve'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('cholesky_solve')
def cholesky_solve(chol, rhs, name=None):
    """Solves systems of linear eqns `A X = RHS`, given Cholesky factorizations.

  Specifically, returns `X` from `A X = RHS`, where `A = L L^T`, `L` is the
  `chol` arg and `RHS` is the `rhs` arg.

  ```python
  # Solve 10 separate 2x2 linear systems:
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 1
  chol = tf.linalg.cholesky(A)  # shape 10 x 2 x 2
  X = tf.linalg.cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
  # tf.matmul(A, X) ~ RHS
  X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

  # Solve five linear systems (K = 5) for every member of the length 10 batch.
  A = ... # shape 10 x 2 x 2
  RHS = ... # shape 10 x 2 x 5
  ...
  X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
  ```

  Args:
    chol:  A `Tensor`.  Must be `float32` or `float64`, shape is `[..., M, M]`.
      Cholesky factorization of `A`, e.g. `chol = tf.linalg.cholesky(A)`.
      For that reason, only the lower triangular parts (including the diagonal)
      of the last two dimensions of `chol` are used.  The strictly upper part is
      assumed to be zero and not accessed.
    rhs:  A `Tensor`, same type as `chol`, shape is `[..., M, K]`.
    name:  A name to give this `Op`.  Defaults to `cholesky_solve`.

  Returns:
    Solution to `A x = rhs`, shape `[..., M, K]`.
  """
    with ops.name_scope(name, 'cholesky_solve', [chol, rhs]):
        y = gen_linalg_ops.matrix_triangular_solve(chol, rhs, adjoint=False, lower=True)
        x = gen_linalg_ops.matrix_triangular_solve(chol, y, adjoint=True, lower=True)
        return x