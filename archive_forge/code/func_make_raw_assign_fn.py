import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
def make_raw_assign_fn(raw_assign_fn, use_handle=True):
    """Wrap `raw_assign_fn` with the proper graph context and device scope.

  Args:
    raw_assign_fn: the function to be wrapped.
    use_handle: if True, the `raw_assign_fn` will be applied to the handle of a
      variable; otherwise it will be applied to the variable itself.

  Returns:
    The wrapped function.
  """

    def assign_fn(var, value, use_locking=False, name=None, read_value=True):
        del use_locking
        handle = var.handle if use_handle else var
        with _maybe_enter_graph(handle), _maybe_on_device(var):
            op = raw_assign_fn(handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
            with ops.control_dependencies([op]):
                if read_value:
                    return var._read_variable_op() if use_handle else var.read_value()
                else:
                    return op
    return assign_fn