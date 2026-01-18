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
def random_tril_matrix(shape, dtype, force_well_conditioned=False, remove_upper=True):
    """[batch] lower triangular matrix.

  Args:
    shape:  `TensorShape` or Python `list`.  Shape of the returned matrix.
    dtype:  `TensorFlow` `dtype` or Python dtype
    force_well_conditioned:  Python `bool`. If `True`, returned matrix will have
      eigenvalues with modulus in `(1, 2)`.  Otherwise, eigenvalues are unit
      normal random variables.
    remove_upper:  Python `bool`.
      If `True`, zero out the strictly upper triangle.
      If `False`, the lower triangle of returned matrix will have desired
      properties, but will not have the strictly upper triangle zero'd out.

  Returns:
    `Tensor` with desired shape and dtype.
  """
    with ops.name_scope('random_tril_matrix'):
        tril = random_normal(shape, dtype=dtype)
        if remove_upper:
            tril = array_ops.matrix_band_part(tril, -1, 0)
        if force_well_conditioned:
            maxval = ops.convert_to_tensor(np.sqrt(2.0), dtype=dtype.real_dtype)
            diag = random_sign_uniform(shape[:-1], dtype=dtype, minval=1.0, maxval=maxval)
            tril = array_ops.matrix_set_diag(tril, diag)
        return tril