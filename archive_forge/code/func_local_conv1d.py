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
@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    """Apply 1D conv with un-shared weights.

  Args:
      inputs: 3D tensor with shape:
          (batch_size, steps, input_dim)
          if data_format is "channels_last" or
          (batch_size, input_dim, steps)
          if data_format is "channels_first".
      kernel: the unshared weight for convolution,
          with shape (output_length, feature_dim, filters).
      kernel_size: a tuple of a single integer,
          specifying the length of the 1D convolution window.
      strides: a tuple of a single integer,
          specifying the stride length of the convolution.
      data_format: the data format, channels_first or channels_last.

  Returns:
      A 3d tensor with shape:
      (batch_size, output_length, filters)
      if data_format='channels_first'
      or 3D tensor with shape:
      (batch_size, filters, output_length)
      if data_format='channels_last'.
  """
    output_shape = (kernel.shape[0],)
    return local_conv(inputs, kernel, kernel_size, strides, output_shape, data_format)