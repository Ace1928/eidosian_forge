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
def assert_is_batch_matrix(tensor):
    """Static assert that `tensor` has rank `2` or higher."""
    sh = tensor.shape
    if sh.ndims is not None and sh.ndims < 2:
        raise ValueError(f'Expected [batch] matrix to have at least two dimensions. Found: {tensor}.')