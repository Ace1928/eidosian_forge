import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def local_conv(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    """Apply N-D convolution with un-shared weights.

  Args:
      inputs: (N+2)-D tensor with shape
          (batch_size, channels_in, d_in1, ..., d_inN)
          if data_format='channels_first', or
          (batch_size, d_in1, ..., d_inN, channels_in)
          if data_format='channels_last'.
      kernel: the unshared weight for N-D convolution,
          with shape (output_items, feature_dim, channels_out), where
          feature_dim = np.prod(kernel_size) * channels_in,
          output_items = np.prod(output_shape).
      kernel_size: a tuple of N integers, specifying the
          spatial dimensions of the N-D convolution window.
      strides: a tuple of N integers, specifying the strides
          of the convolution along the spatial dimensions.
      output_shape: a tuple of (d_out1, ..., d_outN) specifying the spatial
          dimensionality of the output.
      data_format: string, "channels_first" or "channels_last".

  Returns:
      An (N+2)-D tensor with shape:
      (batch_size, channels_out) + output_shape
      if data_format='channels_first', or:
      (batch_size,) + output_shape + (channels_out,)
      if data_format='channels_last'.

  Raises:
      ValueError: if `data_format` is neither
      `channels_last` nor `channels_first`.
  """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))
    kernel_shape = int_shape(kernel)
    feature_dim = kernel_shape[1]
    channels_out = kernel_shape[-1]
    ndims = len(output_shape)
    spatial_dimensions = list(range(ndims))
    xs = []
    output_axes_ticks = [range(axis_max) for axis_max in output_shape]
    for position in itertools.product(*output_axes_ticks):
        slices = [slice(None)]
        if data_format == 'channels_first':
            slices.append(slice(None))
        slices.extend((slice(position[d] * strides[d], position[d] * strides[d] + kernel_size[d]) for d in spatial_dimensions))
        if data_format == 'channels_last':
            slices.append(slice(None))
        xs.append(reshape(inputs[slices], (1, -1, feature_dim)))
    x_aggregate = concatenate(xs, axis=0)
    output = batch_dot(x_aggregate, kernel)
    output = reshape(output, output_shape + (-1, channels_out))
    if data_format == 'channels_first':
        permutation = [ndims, ndims + 1] + spatial_dimensions
    else:
        permutation = [ndims] + spatial_dimensions + [ndims + 1]
    return permute_dimensions(output, permutation)