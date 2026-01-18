import contextlib
from tensorflow.compiler.jit.ops import xla_ops
from tensorflow.compiler.jit.ops import xla_ops_grad  # pylint: disable=unused-import
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import summary_op_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def check_function_argument_count(func, input_arity, infeed_queue):
    """Validate the number of input arguments to an XLA function.

  Args:
    func: the Python function that will be called to generate the body of an XLA
      computation graph.
    input_arity: the number of explicit arguments supplied by the caller.
    infeed_queue: if not None, the infeed queue that will supply
      additional arguments to the function.

  Returns:
    None if function can be called with the supplied number of
      arguments, or an error string if it cannot.
  """

    def format_error(complaint, quantity):
        return '%s %d argument%s' % (complaint, quantity, '' if quantity == 1 else 's')
    num_args_supplied = input_arity
    if infeed_queue is not None:
        num_args_supplied += infeed_queue.number_of_tuple_elements
    arg_spec = tf_inspect.getargspec(func)
    num_func_args = len(arg_spec.args)
    if arg_spec.defaults is None:
        num_func_defaults = 0
    else:
        num_func_defaults = len(arg_spec.defaults)
    min_func_args = num_func_args - num_func_defaults
    if num_args_supplied < min_func_args:
        if num_func_defaults == 0 and arg_spec.varargs is None:
            return format_error('exactly', num_func_args)
        else:
            return format_error('at least', min_func_args)
    if arg_spec.varargs is None and num_args_supplied > num_func_args:
        if num_func_defaults == 0:
            return format_error('exactly', num_func_args)
        else:
            return format_error('at most', num_func_args)
    return None