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
def create_zeros_slot(primary, name, dtype=None, colocate_with_primary=True, *, copy_xla_sharding=False):
    """Create a slot initialized to 0 with same shape as the primary object.

  Args:
    primary: The primary `Variable` or `Tensor`.
    name: Name to use for the slot variable.
    dtype: Type of the slot variable.  Defaults to the type of `primary`.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.
    copy_xla_sharding: Boolean. If True also copies XLA sharding
      from primary.

  Returns:
    A `Variable` object.
  """
    if dtype is None:
        dtype = primary.dtype
    slot_shape = primary.get_shape()
    if slot_shape.is_fully_defined():
        initializer = init_ops.zeros_initializer()
        return create_slot_with_initializer(primary, initializer, slot_shape, dtype, name, colocate_with_primary=colocate_with_primary, copy_xla_sharding=copy_xla_sharding)
    else:
        if isinstance(primary, variables.Variable):
            slot_shape = array_ops.shape(cond.cond(variable_v1.is_variable_initialized(primary), primary.read_value, lambda: primary.initial_value))
        else:
            slot_shape = array_ops.shape(primary)
        val = array_ops.zeros(slot_shape, dtype=dtype)
        return create_slot(primary, val, name, colocate_with_primary=colocate_with_primary, copy_xla_sharding=copy_xla_sharding)