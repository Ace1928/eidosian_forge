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
@tf_export(v1=['nn.crelu'])
@dispatch.add_dispatch_support
def crelu(features, name=None, axis=-1):
    """Computes Concatenated ReLU.

  Concatenates a ReLU which selects only the positive part of the activation
  with a ReLU which selects only the *negative* part of the activation.
  Note that as a result this non-linearity doubles the depth of the activations.
  Source: [Understanding and Improving Convolutional Neural Networks via
  Concatenated Rectified Linear Units. W. Shang, et
  al.](https://arxiv.org/abs/1603.05201)

  Args:
    features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
      `int16`, or `int8`.
    name: A name for the operation (optional).
    axis: The axis that the output values are concatenated along. Default is -1.

  Returns:
    A `Tensor` with the same type as `features`.

  References:
    Understanding and Improving Convolutional Neural Networks via Concatenated
    Rectified Linear Units:
      [Shang et al., 2016](http://proceedings.mlr.press/v48/shang16)
      ([pdf](http://proceedings.mlr.press/v48/shang16.pdf))
  """
    with ops.name_scope(name, 'CRelu', [features]) as name:
        features = ops.convert_to_tensor(features, name='features')
        c = array_ops.concat([features, -features], axis, name=name)
        return gen_nn_ops.relu(c)