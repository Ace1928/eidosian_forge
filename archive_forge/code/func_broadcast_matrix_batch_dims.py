import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def broadcast_matrix_batch_dims(batch_matrices, name=None):
    """Broadcast leading dimensions of zero or more [batch] matrices.

  Example broadcasting one batch dim of two simple matrices.

  ```python
  x = [[1, 2],
       [3, 4]]  # Shape [2, 2], no batch dims

  y = [[[1]]]   # Shape [1, 1, 1], 1 batch dim of shape [1]

  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc
  ==> [[[1, 2],
        [3, 4]]]  # Shape [1, 2, 2], 1 batch dim of shape [1].

  y_bc
  ==> same as y
  ```

  Example broadcasting many batch dims

  ```python
  x = tf.random.normal(shape=(2, 3, 1, 4, 4))
  y = tf.random.normal(shape=(1, 3, 2, 5, 5))
  x_bc, y_bc = broadcast_matrix_batch_dims([x, y])

  x_bc.shape
  ==> (2, 3, 2, 4, 4)

  y_bc.shape
  ==> (2, 3, 2, 5, 5)
  ```

  Args:
    batch_matrices:  Iterable of `Tensor`s, each having two or more dimensions.
    name:  A string name to prepend to created ops.

  Returns:
    bcast_matrices: List of `Tensor`s, with `bcast_matrices[i]` containing
      the values from `batch_matrices[i]`, with possibly broadcast batch dims.

  Raises:
    ValueError:  If any input `Tensor` is statically determined to have less
      than two dimensions.
  """
    with ops.name_scope(name or 'broadcast_matrix_batch_dims', values=batch_matrices):
        check_ops.assert_proper_iterable(batch_matrices)
        batch_matrices = list(batch_matrices)
        for i, mat in enumerate(batch_matrices):
            batch_matrices[i] = tensor_conversion.convert_to_tensor_v2_with_dispatch(mat)
            assert_is_batch_matrix(batch_matrices[i])
        if len(batch_matrices) < 2:
            return batch_matrices
        bcast_batch_shape = batch_matrices[0].shape[:-2]
        for mat in batch_matrices[1:]:
            bcast_batch_shape = array_ops.broadcast_static_shape(bcast_batch_shape, mat.shape[:-2])
        if bcast_batch_shape.is_fully_defined():
            for i, mat in enumerate(batch_matrices):
                if mat.shape[:-2] != bcast_batch_shape:
                    bcast_shape = array_ops.concat([bcast_batch_shape.as_list(), array_ops.shape(mat)[-2:]], axis=0)
                    batch_matrices[i] = array_ops.broadcast_to(mat, bcast_shape)
            return batch_matrices
        bcast_batch_shape = array_ops.shape(batch_matrices[0])[:-2]
        for mat in batch_matrices[1:]:
            bcast_batch_shape = array_ops.broadcast_dynamic_shape(bcast_batch_shape, array_ops.shape(mat)[:-2])
        for i, mat in enumerate(batch_matrices):
            batch_matrices[i] = array_ops.broadcast_to(mat, array_ops.concat([bcast_batch_shape, array_ops.shape(mat)[-2:]], axis=0))
        return batch_matrices