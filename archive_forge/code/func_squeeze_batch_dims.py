import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def squeeze_batch_dims(inp, op, inner_rank):
    """Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.

  Where `squeeze_batch` reshapes `inp` to shape
  `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
  and `unsqueeze_batch` does the reverse reshape but on the output.

  Args:
    inp: A tensor with dims `batch_shape + inner_shape` where `inner_shape`
      is length `inner_rank`.
    op: A callable that takes a single input tensor and returns a single.
      output tensor.
    inner_rank: A python integer.

  Returns:
    `unsqueeze_batch_op(squeeze_batch(inp))`.
  """
    with ops.name_scope_v2('squeeze_batch_dims'):
        shape = inp.shape
        inner_shape = shape[-inner_rank:]
        if not inner_shape.is_fully_defined():
            inner_shape = array_ops.shape(inp)[-inner_rank:]
        batch_shape = shape[:-inner_rank]
        if not batch_shape.is_fully_defined():
            batch_shape = array_ops.shape(inp)[:-inner_rank]
        if isinstance(inner_shape, tensor_shape.TensorShape):
            inp_reshaped = array_ops.reshape(inp, [-1] + inner_shape.as_list())
        else:
            inp_reshaped = array_ops.reshape(inp, array_ops.concat(([-1], inner_shape), axis=-1))
        out_reshaped = op(inp_reshaped)
        out_inner_shape = out_reshaped.shape[-inner_rank:]
        if not out_inner_shape.is_fully_defined():
            out_inner_shape = array_ops.shape(out_reshaped)[-inner_rank:]
        out = array_ops.reshape(out_reshaped, array_ops.concat((batch_shape, out_inner_shape), axis=-1))
        out.set_shape(inp.shape[:-inner_rank] + out.shape[-inner_rank:])
        return out