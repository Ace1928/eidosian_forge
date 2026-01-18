from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
def supports_default_grad(t):
    """Whether tensor `t` supports creating a default gradient.

  This function assumes that `t` is of a trainable type.

  Args:
    t: Tensor

  Returns:
    Bool
  """
    if t.dtype == dtypes.resource:
        handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
        if handle_data is None or not handle_data.is_set or len(handle_data.shape_and_type) != 1:
            return False
    return True