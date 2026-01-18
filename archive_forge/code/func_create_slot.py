from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
def create_slot(primary, val, name, colocate_with_primary=True, *, copy_xla_sharding=False):
    """Create a slot initialized to the given value.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    val: A `Tensor` specifying the initial value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.
    copy_xla_sharding: Boolean. If True also copies XLA sharding
      from primary.

  Returns:
    A `Variable` object.
  """
    validate_shape = val.get_shape().is_fully_defined()
    if isinstance(primary, variables.Variable):
        prefix = primary._shared_name
    else:
        prefix = primary.op.name
    with variable_scope.variable_scope(None, prefix + '/' + name):
        if colocate_with_primary:
            distribution_strategy = distribute_lib.get_strategy()
            with distribution_strategy.extended.colocate_vars_with(primary):
                return _create_slot_var(primary, val, '', validate_shape, None, None, copy_xla_sharding=copy_xla_sharding)
        else:
            return _create_slot_var(primary, val, '', validate_shape, None, None, copy_xla_sharding=copy_xla_sharding)