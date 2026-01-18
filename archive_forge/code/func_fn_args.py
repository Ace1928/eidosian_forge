import functools
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.module import module
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
def fn_args(fn):
    """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
    if isinstance(fn, functools.partial):
        args = fn_args(fn.func)
        args = [a for a in args[len(fn.args):] if a not in (fn.keywords or [])]
    else:
        if hasattr(fn, '__call__') and tf_inspect.ismethod(fn.__call__):
            fn = fn.__call__
        args = tf_inspect.getfullargspec(fn).args
        if _is_bound_method(fn) and args:
            args.pop(0)
    return tuple(args)