import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def get_item(target, i, opts):
    """The slice read operator (i.e. __getitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports getitem semantics.
    i: Index to read from.
    opts: A GetItemOpts object.

  Returns:
    The read element.

  Raises:
    ValueError: if target is not of a supported type.
  """
    assert isinstance(opts, GetItemOpts)
    if isinstance(target, tensor_array_ops.TensorArray):
        return _tf_tensorarray_get_item(target, i)
    elif tensor_util.is_tf_type(target):
        if target.dtype == dtypes.variant:
            return _tf_tensor_list_get_item(target, i, opts)
        elif target.dtype == dtypes.string and target.shape.ndims == 0:
            return _tf_tensor_string_get_item(target, i)
        else:
            return _tf_tensor_get_item(target, i)
    else:
        return _py_get_item(target, i)