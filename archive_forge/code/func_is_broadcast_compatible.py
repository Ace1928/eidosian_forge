import itertools
from tensorflow.python.framework import tensor_shape
def is_broadcast_compatible(shape_x, shape_y):
    """Returns True if `shape_x` and `shape_y` are broadcast compatible.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted
    to.  False otherwise.
  """
    if shape_x.ndims is None or shape_y.ndims is None:
        return False
    return _broadcast_shape_helper(shape_x, shape_y) is not None