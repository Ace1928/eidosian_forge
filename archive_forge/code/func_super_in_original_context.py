import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def super_in_original_context(f, args, caller_fn_scope):
    """Executes the super function in the context of a specified function.

  See https://docs.python.org/3/library/functions.html#super for the exact
  details

  Args:
    f: Callable, typically the super builtin
    args: List[Any], the original call arguments
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made

  Returns:
    The result of calling `f` as if it was called in the frame indicated by
      `caller_fn_scope`.
  """
    if args:
        return f(*args)
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)
    type_arg = ctx_frame.f_locals['__class__']
    self_arg_name = ctx_frame.f_code.co_varnames[0]
    self_arg = ctx_frame.f_locals[self_arg_name]
    return f(type_arg, self_arg)