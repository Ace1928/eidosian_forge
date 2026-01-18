from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['saved_model.main_op.main_op'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op():
    """Returns a main op to init variables and tables.

  Returns the main op including the group of ops that initializes all
  variables, initializes local variables and initialize all tables.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
    init = variables.global_variables_initializer()
    init_local = variables.local_variables_initializer()
    init_tables = lookup_ops.tables_initializer()
    return control_flow_ops.group(init, init_local, init_tables)