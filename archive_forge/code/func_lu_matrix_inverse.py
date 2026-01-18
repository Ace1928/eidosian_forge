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
@tf_export('linalg.lu_matrix_inverse')
@dispatch.add_dispatch_support
def lu_matrix_inverse(lower_upper, perm, validate_args=False, name=None):
    """Computes the inverse given the LU decomposition(s) of one or more matrices.

  This op is conceptually identical to,

  ```python
  inv_X = tf.lu_matrix_inverse(*tf.linalg.lu(X))
  tf.assert_near(tf.matrix_inverse(X), inv_X)
  # ==> True
  ```

  Note: this function does not verify the implied matrix is actually invertible
  nor is this condition checked even when `validate_args=True`.

  Args:
    lower_upper: `lu` as returned by `tf.linalg.lu`, i.e., if `matmul(P,
      matmul(L, U)) = X` then `lower_upper = L + U - eye`.
    perm: `p` as returned by `tf.linag.lu`, i.e., if `matmul(P, matmul(L, U)) =
      X` then `perm = argmax(P)`.
    validate_args: Python `bool` indicating whether arguments should be checked
      for correctness. Note: this function does not verify the implied matrix is
        actually invertible, even when `validate_args=True`.
      Default value: `False` (i.e., don't validate arguments).
    name: Python `str` name given to ops managed by this object.
      Default value: `None` (i.e., 'lu_matrix_inverse').

  Returns:
    inv_x: The matrix_inv, i.e.,
      `tf.matrix_inverse(tf.linalg.lu_reconstruct(lu, perm))`.

  #### Examples

  ```python
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp

  x = [[[3., 4], [1, 2]],
       [[7., 8], [3, 4]]]
  inv_x = tf.linalg.lu_matrix_inverse(*tf.linalg.lu(x))
  tf.assert_near(tf.matrix_inverse(x), inv_x)
  # ==> True
  ```

  """
    with ops.name_scope(name or 'lu_matrix_inverse'):
        lower_upper = ops.convert_to_tensor(lower_upper, dtype_hint=dtypes.float32, name='lower_upper')
        perm = ops.convert_to_tensor(perm, dtype_hint=dtypes.int32, name='perm')
        assertions = lu_reconstruct_assertions(lower_upper, perm, validate_args)
        if assertions:
            with ops.control_dependencies(assertions):
                lower_upper = array_ops.identity(lower_upper)
                perm = array_ops.identity(perm)
        shape = array_ops.shape(lower_upper)
        return lu_solve(lower_upper, perm, rhs=eye(shape[-1], batch_shape=shape[:-2], dtype=lower_upper.dtype), validate_args=False)