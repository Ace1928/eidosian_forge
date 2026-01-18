import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def list_stack(list_, opts):
    """The list stack function.

  This does not have a direct correspondent in Python. The closest idiom to
  this is tf.append or np.stack. It's different from those in the sense that it
  accepts a Tensor list, rather than a list of tensors. It can also accept
  TensorArray. When the target is anything else, the dispatcher will rely on
  ctx.original_call for fallback.

  Args:
    list_: An entity that supports append semantics.
    opts: A ListStackOpts object.

  Returns:
    The output of the stack operation, typically a Tensor.
  """
    assert isinstance(opts, ListStackOpts)
    if isinstance(list_, tensor_array_ops.TensorArray):
        return _tf_tensorarray_stack(list_)
    elif tensor_util.is_tf_type(list_):
        if list_.dtype == dtypes.variant:
            return _tf_tensor_list_stack(list_, opts)
        else:
            return list_
    else:
        return _py_list_stack(list_, opts)