import functools
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def op_list_to_dict(op_list, convert_variable_to_tensor=True):
    """Create a dictionary of names to operation lists.

  This method is only used when the variable name matters (e.g. when saving
  or restoring from a TF1 name-based checkpoint). In TF2, this can be called
  from `tf.train.Checkpoint.restore` when loading from a name-based checkpoint.

  Args:
    op_list: A (nested) list, tuple, or set of Variables or SaveableObjects.
    convert_variable_to_tensor: Whether or not to convert single Variables
      with no slice info into Tensors.

  Returns:
    A dictionary of names to the operations that must be saved under
    that name.  Variables with save_slice_info are grouped together under the
    same key in no particular order.

  Raises:
    TypeError: If the type of op_list or its elements is not supported.
    ValueError: If at least two saveables share the same name.
  """
    if not isinstance(op_list, (list, tuple, set)):
        raise TypeError(f'Variables to save should be passed in a dict or a list. Got {op_list}')
    op_list = nest.flatten(list(op_list))
    op_list = sorted(op_list, key=lambda x: x.name)
    names_to_saveables = {}
    for var in op_list:
        resource_or_ref_variable = isinstance(var, resource_variable_ops.BaseResourceVariable) or isinstance(var, ref_variable.RefVariable)
        if isinstance(var, saveable_object.SaveableObject):
            names_to_saveables[var.name] = var
        elif isinstance(var, variables.PartitionedVariable):
            if var.name in names_to_saveables:
                raise ValueError(f'At least two variables have the same name: {var.name}')
            names_to_saveables[var.name] = var
        elif isinstance(var, variables.Variable) and var._save_slice_info:
            name = var._save_slice_info.full_name
            if name in names_to_saveables:
                if not isinstance(names_to_saveables[name], list):
                    raise ValueError(f'Mixing slices and non-slices with the same name: {name}')
                names_to_saveables[name].append(var)
            else:
                names_to_saveables[name] = [var]
        elif isinstance(var, trackable.Trackable) and (not resource_or_ref_variable):
            trackable_saveables = [factory() if callable(factory) else factory for factory in saveable_objects_from_trackable(var, tf1_saver=True).values()]
            names_to_saveables.update(op_list_to_dict(trackable_saveables))
        elif not getattr(var, '_in_graph_mode', True):
            if not isinstance(var, resource_variable_ops.BaseResourceVariable):
                raise ValueError(f'Can only save/restore ResourceVariables when eager execution is enabled. Got type: {type(var)}.')
            set_var = names_to_saveables.setdefault(var._shared_name, var)
            if set_var is not var:
                raise ValueError(f"Two different ResourceVariable objects with the same shared_name '{var._shared_name}' were passed to the Saver. This likely means that they were created in different Graphs or isolated contexts, and may not be checkpointed together.")
        else:
            if convert_variable_to_tensor:
                if isinstance(var, resource_variable_ops.BaseResourceVariable):
                    var = var._graph_element
                else:
                    var = ops.convert_to_tensor(var, as_ref=True)
                if not _tensor_comes_from_variable(var):
                    raise TypeError(f'Variable to save is not a Variable: {var}')
            if var.op.type == 'ReadVariableOp':
                name = var.op.inputs[0].op.name
            else:
                name = var.op.name
            if name in names_to_saveables:
                raise ValueError(f'At least two variables have the same name: {name}')
            names_to_saveables[name] = var
    return names_to_saveables