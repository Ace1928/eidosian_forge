import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def convert_padding(padding, expected_length=4):
    """Converts Python padding to C++ padding for ops which take EXPLICIT padding.

  Args:
    padding: the `padding` argument for a Python op which supports EXPLICIT
      padding.
    expected_length: Expected number of entries in the padding list when
      explicit padding is used.

  Returns:
    (padding, explicit_paddings) pair, which should be passed as attributes to a
    C++ op.

  Raises:
    ValueError: If padding is invalid.
  """
    explicit_paddings = []
    if padding == 'EXPLICIT':
        raise ValueError("'EXPLICIT' is not a valid value for `padding`. To use explicit padding, `padding` must be a list.")
    if isinstance(padding, (list, tuple)):
        for i, dim_paddings in enumerate(padding):
            if not isinstance(dim_paddings, (list, tuple)):
                raise ValueError(f'When `padding` is a list, each element of `padding` must be a list/tuple of size 2. Received: padding={padding} with element at index {i} of type {type(dim_paddings)}')
            if len(dim_paddings) != 2:
                raise ValueError(f'When `padding` is a list, each element of `padding` must be a list/tuple of size 2. Received: padding={padding} with element at index {i} of size {len(dim_paddings)}')
            explicit_paddings.extend(dim_paddings)
        if len(padding) != expected_length:
            raise ValueError(f'When padding is a list, it must be of size {expected_length}. Received: padding={padding} of size {len(padding)}')
        padding = 'EXPLICIT'
    return (padding, explicit_paddings)