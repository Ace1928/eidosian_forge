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
def convolution_internal(input, filters, strides=None, padding='VALID', data_format=None, dilations=None, name=None, call_from_convolution=True, num_spatial_dims=None):
    """Internal function which performs rank agnostic convolution.

  Args:
    input: See `convolution`.
    filters: See `convolution`.
    strides: See `convolution`.
    padding: See `convolution`.
    data_format: See `convolution`.
    dilations: See `convolution`.
    name: See `convolution`.
    call_from_convolution: See `convolution`.
    num_spatial_dims: (Optional.).  It is a integer describing the
      rank of the spatial dimensions.  For `1-D`, `2-D` and `3-D` convolutions,
      the value of `num_spatial_dims` is `1`, `2`, and `3`, respectively.
      This argument is only required to disambiguate the rank of `batch_shape`
      when `filter_shape.ndims is None` and `len(batch_shape) > 1`.  For
      backwards compatibility, if `num_spatial_dims is None` and
     `filter_shape.ndims is None`, then `len(batch_shape)` is assumed to be
     `1` (i.e., the input is expected to be
     `[batch_size, num_channels] + input_spatial_shape`
     or `[batch_size] + input_spatial_shape + [num_channels]`.

  Returns:
    A tensor of shape and dtype matching that of `input`.

  Raises:
    ValueError: If input and filter both have unknown shapes, or if
      `num_spatial_dims` is provided and incompatible with the value
      estimated from `filters.shape`.
  """
    if not isinstance(filters, variables_lib.Variable) and (not tensor_util.is_tf_type(filters)):
        with ops.name_scope('convolution_internal', None, [filters, input]):
            filters = ops.convert_to_tensor(filters, name='filters')
    if not isinstance(input, tensor_lib.Tensor) and (not tensor_util.is_tf_type(input)):
        with ops.name_scope('convolution_internal', None, [filters, input]):
            input = ops.convert_to_tensor(input, name='input')
    filters_rank = filters.shape.rank
    inputs_rank = input.shape.rank
    if num_spatial_dims is None:
        if filters_rank:
            num_spatial_dims = filters_rank - 2
        elif inputs_rank:
            num_spatial_dims = inputs_rank - 2
        else:
            raise ValueError(f'When `num_spatial_dims` is not set, one of `input.shape.rank` or `filters.shape.rank` must be known. Received: input.shape={input.shape} of rank {inputs_rank} and filters.shape={filters.shape} of rank {filters_rank}')
    elif filters_rank and filters_rank - 2 != num_spatial_dims:
        raise ValueError(f'`filters.shape.rank - 2` should equal `num_spatial_dims`. Received: filters.shape={filters.shape} of rank {filters_rank} and num_spatial_dims={num_spatial_dims}')
    if inputs_rank:
        num_batch_dims = inputs_rank - num_spatial_dims - 1
    else:
        num_batch_dims = 1
    if num_spatial_dims not in {1, 2, 3}:
        raise ValueError(f'`num_spatial_dims` must be 1, 2, or 3. Received: num_spatial_dims={num_spatial_dims}.')
    if data_format is None or data_format in _CHANNELS_LAST_FORMATS:
        channel_index = num_batch_dims + num_spatial_dims
    else:
        channel_index = num_batch_dims
    if dilations is None:
        dilations = _get_sequence(dilations, num_spatial_dims, channel_index, 'dilations')
        is_dilated_conv = False
    else:
        dilations = _get_sequence(dilations, num_spatial_dims, channel_index, 'dilations')
        is_dilated_conv = any((i != 1 for i in dilations))
    strides = _get_sequence(strides, num_spatial_dims, channel_index, 'strides')
    has_tpu_context = device_context.enclosing_tpu_context() is not None
    if name:
        default_name = None
    elif not has_tpu_context or call_from_convolution:
        default_name = 'convolution'
    elif num_spatial_dims == 2:
        default_name = 'Conv2D'
    elif num_spatial_dims == 3:
        default_name = 'Conv3D'
    else:
        default_name = 'conv1d'
    with ops.name_scope(name, default_name, [input, filters]) as name:
        if not is_dilated_conv or has_tpu_context:
            if num_spatial_dims == 2:
                op = _conv2d_expanded_batch
            elif num_spatial_dims == 3:
                op = _conv3d_expanded_batch
            else:
                op = conv1d
            return op(input, filters, strides, padding=padding, data_format=data_format, dilations=dilations, name=name)
        else:
            if channel_index == 1:
                strides = strides[2:]
                dilations = dilations[2:]
            else:
                strides = strides[1:-1]
                dilations = dilations[1:-1]
            op = Convolution(tensor_shape.as_shape(input.shape), tensor_shape.as_shape(filters.shape), padding, strides=strides, dilation_rate=dilations, name=name, data_format=data_format, num_spatial_dims=num_spatial_dims)
            return op(input, filters)