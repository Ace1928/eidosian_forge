import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def random_positive_definite_matrix(shape, dtype, oversampling_ratio=4, force_well_conditioned=False):
    """[batch] positive definite Wisart matrix.

  A Wishart(N, S) matrix is the S sample covariance matrix of an N-variate
  (standard) Normal random variable.

  Args:
    shape:  `TensorShape` or Python list.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype.
    oversampling_ratio: S / N in the above.  If S < N, the matrix will be
      singular (unless `force_well_conditioned is True`).
    force_well_conditioned:  Python bool.  If `True`, add `1` to the diagonal
      of the Wishart matrix, then divide by 2, ensuring most eigenvalues are
      close to 1.

  Returns:
    `Tensor` with desired shape and dtype.
  """
    dtype = dtypes.as_dtype(dtype)
    if not tensor_util.is_tf_type(shape):
        shape = tensor_shape.TensorShape(shape)
        shape.dims[-1].assert_is_compatible_with(shape.dims[-2])
    shape = shape.as_list()
    n = shape[-2]
    s = oversampling_ratio * shape[-1]
    wigner_shape = shape[:-2] + [n, s]
    with ops.name_scope('random_positive_definite_matrix'):
        wigner = random_normal(wigner_shape, dtype=dtype, stddev=math_ops.cast(1 / np.sqrt(s), dtype.real_dtype))
        wishart = math_ops.matmul(wigner, wigner, adjoint_b=True)
        if force_well_conditioned:
            wishart += linalg_ops.eye(n, dtype=dtype)
            wishart /= math_ops.cast(2, dtype)
        return wishart