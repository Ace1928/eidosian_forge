import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
def all_sum(tensors):
    """Returns a list of tensors with the all-reduce sum across `tensors`.

  The computation is done with an all-reduce operation, so if only some of the
  returned tensors are evaluated then the computation will hang.

  Args:
    tensors: The input tensors across which to sum; must be assigned
      to GPU devices.

  Returns:
    List of tensors, each with the sum of the input tensors, where tensor i has
    the same device as `tensors[i]`.
  """
    return _apply_all_reduce('sum', tensors)