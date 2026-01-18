from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object
def is_saving_non_distributed():
    """Returns whether we're saving a non-distributed version of the model.

  It returns True iff we are in saving context and are saving a non-distributed
  version of the model. That is, SaveOptions.experimental_variable_policy is
  NONE.

  Returns:
    A boolean.
  """
    if not save_context.in_save_context():
        return False
    options = save_context.get_save_options()
    return options.experimental_variable_policy != save_options.VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES