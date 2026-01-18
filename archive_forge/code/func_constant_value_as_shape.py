import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def constant_value_as_shape(tensor):
    """A version of `constant_value()` that returns a `TensorShape`.

  This version should be used when a constant tensor value is
  interpreted as a (possibly partial) shape, e.g. in the shape
  function for `tf.reshape()`. By explicitly requesting a
  `TensorShape` as the return value, it is possible to represent
  unknown dimensions; by contrast, `constant_value()` is
  all-or-nothing.

  Args:
    tensor: The rank-0 or rank-1 Tensor to be evaluated.

  Returns:
    A `TensorShape` based on the constant value of the given `tensor`.

  Raises:
    ValueError: If the shape is rank-0 and is not statically known to be -1.
  """
    if isinstance(tensor, core.Value):
        return tensor_shape.TensorShape([dim if dim != -1 else None for dim in tensor.numpy()])
    if tensor.get_shape().ndims == 0:
        value = constant_value(tensor)
        if value is None:
            raise ValueError("Received a scalar with unknown value as shape; require a statically known scalar with value '-1' to describe an unknown shape.")
        if value != -1:
            raise ValueError(f"Received a scalar value '{value}' as shape; require a statically known scalar with value '-1' to describe an unknown shape.")
        return tensor_shape.unknown_shape()
    shape = tensor.get_shape().with_rank(1)
    if shape == [0]:
        return tensor_shape.TensorShape([])
    elif tensor.op.type == 'Cast':
        pre_cast = constant_value_as_shape(tensor.op.inputs[0])
        if pre_cast.dims is None:
            return pre_cast
        cast_dtype = dtypes.as_dtype(tensor.op.get_attr('DstT'))
        if cast_dtype not in (dtypes.int32, dtypes.int64):
            return tensor_shape.unknown_shape(shape.dims[0].value)
        dest_dtype_shape_array = np.array([x if x is not None else -1 for x in pre_cast.as_list()]).astype(cast_dtype.as_numpy_dtype)
        return tensor_shape.TensorShape([x if x >= 0 else None for x in dest_dtype_shape_array])
    elif tensor.op.type == 'Shape':
        return tensor.op.inputs[0].get_shape()
    elif tensor.op.type == 'Pack':
        ret = tensor_shape.TensorShape([])
        assert tensor.op.get_attr('axis') == 0
        for pack_input in tensor.op.inputs:
            pack_input_val = constant_value(pack_input)
            if pack_input_val is None or pack_input_val < 0:
                new_dim = tensor_shape.Dimension(None)
            else:
                new_dim = tensor_shape.Dimension(pack_input_val)
            ret = ret.concatenate([new_dim])
        return ret
    elif tensor.op.type == 'Concat':
        ret = tensor_shape.TensorShape([])
        for concat_input in tensor.op.inputs[1:]:
            ret = ret.concatenate(constant_value_as_shape(concat_input))
        return ret
    elif tensor.op.type == 'ConcatV2':
        ret = tensor_shape.TensorShape([])
        for concat_input in tensor.op.inputs[:-1]:
            ret = ret.concatenate(constant_value_as_shape(concat_input))
        return ret
    elif tensor.op.type == 'StridedSlice':
        try:
            begin = constant_value(tensor.op.inputs[1])
            end = constant_value(tensor.op.inputs[2])
            strides = constant_value(tensor.op.inputs[3])
            if begin is not None and end is not None and (strides is not None):
                begin = begin[0]
                end = end[0]
                strides = strides[0]
                begin_mask = tensor.op.get_attr('begin_mask')
                if begin_mask == 1:
                    begin = None
                end_mask = tensor.op.get_attr('end_mask')
                if end_mask == 1:
                    end = None
                ellipsis_mask = tensor.op.get_attr('ellipsis_mask')
                new_axis_mask = tensor.op.get_attr('new_axis_mask')
                shrink_axis_mask = tensor.op.get_attr('shrink_axis_mask')
                valid_attributes = not ellipsis_mask and (not new_axis_mask) and (not shrink_axis_mask) and (not begin_mask or begin_mask == 1) and (not end_mask or end_mask == 1)
                if valid_attributes:
                    prev = constant_value_as_shape(tensor.op.inputs[0])
                    prev = prev[begin:end:strides]
                    ret = tensor_shape.TensorShape(prev)
                    return ret
        except ValueError:
            pass
        except TypeError:
            pass
    elif tensor.op.type == 'Placeholder' and tensor.op.graph.building_function and hasattr(tensor.op.graph, 'internal_captures'):
        for i, capture in enumerate(tensor.op.graph.internal_captures):
            if capture is tensor:
                external_capture = tensor.op.graph.external_captures[i]
                return constant_value_as_shape(external_capture)
    ret = tensor_shape.unknown_shape(shape.dims[0].value)
    value = constant_value(tensor)
    if value is not None:
        ret = ret.merge_with(tensor_shape.TensorShape([d if d >= 0 else None for d in value]))
    return ret