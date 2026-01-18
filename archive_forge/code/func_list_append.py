import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def list_append(list_, x):
    """The list append function.

  Note: it is unspecified where list_ will be mutated or not. If list_ is
  a TensorFlow entity, it will not be typically mutated. If list_ is a plain
  list, it will be. In general, if the list is mutated then the return value
  should point to the original entity.

  Args:
    list_: An entity that supports append semantics.
    x: The element to append.

  Returns:
    Same as list_, after the append was performed.

  Raises:
    ValueError: if list_ is not of a known list-like type.
  """
    if isinstance(list_, tensor_array_ops.TensorArray):
        return _tf_tensorarray_append(list_, x)
    elif tensor_util.is_tf_type(list_):
        if list_.dtype == dtypes.variant:
            return _tf_tensor_list_append(list_, x)
        else:
            raise ValueError('tensor lists are expected to be Tensors with dtype=tf.variant, instead found %s' % list_)
    else:
        return _py_list_append(list_, x)