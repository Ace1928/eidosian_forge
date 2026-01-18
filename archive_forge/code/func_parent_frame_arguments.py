import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def parent_frame_arguments():
    """Returns parent frame arguments.

  When called inside a function, returns a dictionary with the caller's function
  arguments. These are positional arguments and keyword arguments (**kwargs),
  while variable arguments (*varargs) are excluded.

  When called at global scope, this will return an empty dictionary, since there
  are no arguments.

  WARNING: If caller function argument names are overloaded before invoking
  this method, then values will reflect the overloaded value. For this reason,
  we recommend calling `parent_frame_arguments` at the beginning of the
  function.
  """
    arg_names, variable_arg_name, keyword_arg_name, local_vars = tf_inspect._inspect.getargvalues(tf_inspect._inspect.stack()[1][0])
    local_vars.pop(variable_arg_name, {})
    keyword_args = local_vars.pop(keyword_arg_name, {})
    final_args = {}
    for arg_name in arg_names:
        final_args[arg_name] = local_vars.pop(arg_name)
    final_args.update(keyword_args)
    return final_args