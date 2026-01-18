from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
def alias_tensors(*args):
    """Wraps any Tensor arguments with an identity op.

  Any other argument, including Variables, is returned unchanged.

  Args:
    *args: Any arguments. Must contain at least one element.

  Returns:
    Same as *args, with Tensor instances replaced as described.

  Raises:
    ValueError: If args doesn't meet the requirements.
  """

    def alias_if_tensor(a):
        return array_ops.identity(a) if isinstance(a, tensor.Tensor) else a
    if len(args) > 1:
        return (alias_if_tensor(a) for a in args)
    elif len(args) == 1:
        return alias_if_tensor(args[0])
    raise ValueError('at least one argument required')