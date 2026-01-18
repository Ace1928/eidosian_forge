import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def build_nccl_all_reduce(input_tensors, red_op, un_op=None):
    """Build a subgraph that does one full all-reduce, using NCCL.

  Args:
    input_tensors: list of `tf.Tensor` of same-shape and type values to
      be reduced.
    red_op: binary elementwise reduction operator. Must be one of
      {tf.add}
    un_op: optional unary elementwise Op to apply to fully-reduce values.

  Returns:
    list of `tf.Tensor` of reduced values.

  Raises:
    ValueError: red_op not supported.
  """
    if red_op == math_ops.add:
        output_tensors = nccl_ops.all_sum(input_tensors)
    else:
        raise ValueError('red_op not supported by NCCL all-reduce: ', red_op)
    if un_op:
        un_op_wrapped = []
        for t in output_tensors:
            with ops.colocate_with(t):
                un_op_wrapped.append(un_op(t))
        output_tensors = un_op_wrapped
    return output_tensors