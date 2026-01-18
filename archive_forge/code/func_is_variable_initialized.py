from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['is_variable_initialized'])
@tf_should_use.should_use_result
def is_variable_initialized(variable):
    """Tests if a variable has been initialized.

  Args:
    variable: A `Variable`.

  Returns:
    Returns a scalar boolean Tensor, `True` if the variable has been
    initialized, `False` otherwise.
  """
    return state_ops.is_variable_initialized(variable)