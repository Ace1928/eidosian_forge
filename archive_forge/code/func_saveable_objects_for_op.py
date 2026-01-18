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
def saveable_objects_for_op(op, name):
    """Create `SaveableObject`s from an operation.

  Args:
    op: A variable, operation, or SaveableObject to coerce into a
      SaveableObject.
    name: A string name for the SaveableObject.

  Yields:
    `SaveableObject`s which together save/restore `op`.

  Raises:
    TypeError: If `name` is not a string.
    ValueError: For operations with no known conversion to SaveableObject.
  """
    if not isinstance(name, str):
        raise TypeError(f'names_to_saveables must be a dict mapping string names to trackable operations. Name is not a string: {name}')
    if isinstance(op, saveable_object.SaveableObject):
        yield op
    elif isinstance(op, (list, tuple, variables.PartitionedVariable)):
        if isinstance(op, variables.PartitionedVariable):
            op = list(op)
        slice_name = None
        for variable in op:
            if isinstance(variable, saveable_object.SaveableObject):
                yield variable
                continue
            if not isinstance(variable, variables.Variable):
                raise ValueError(f'Slices must all be Variables: {variable}')
            if not variable._save_slice_info:
                raise ValueError(f'Slices must all be slices: {variable}')
            if slice_name is None:
                slice_name = variable._save_slice_info.full_name
            elif slice_name != variable._save_slice_info.full_name:
                raise ValueError(f'Slices must all be from the same tensor: {slice_name} != {variable._save_slice_info.full_name}')
            if variable.op.type in _REF_VARIABLE_OPS:
                yield ReferenceVariableSaveable(variable, variable._save_slice_info.spec, name)
            else:
                yield ResourceVariableSaveable(variable, variable._save_slice_info.spec, name)
    elif isinstance(op, trackable.Trackable) and (not isinstance(op, variables.Variable)):
        for attr, factory in saveable_objects_from_trackable(op, tf1_saver=True).items():
            if attr == trackable.VARIABLE_VALUE_KEY:
                full_name = name
            elif attr == trackable_utils.SERIALIZE_TO_TENSORS_NAME:
                full_name = name
            else:
                full_name = name + '_' + attr
            op = factory(full_name) if callable(factory) else factory
            for op in saveable_objects_for_op(op, op.name):
                yield op
    elif isinstance(op, resource_variable_ops.BaseResourceVariable):
        if op._in_graph_mode:
            variable = op._graph_element
        else:
            variable = op
        yield ResourceVariableSaveable(variable, '', name)
    else:
        if context.executing_eagerly():
            raise ValueError(f'Can only save/restore ResourceVariables when executing eagerly, got type: {type(op)}.')
        variable = ops.convert_to_tensor(op, as_ref=True)
        if not _tensor_comes_from_variable(variable):
            raise TypeError(f'names_to_saveables must be a dict mapping string names to Tensors/Variables. Not a variable: {variable}')
        if variable.op.type in _REF_VARIABLE_OPS:
            yield ReferenceVariableSaveable(variable, '', name)
        else:
            yield ResourceVariableSaveable(variable, '', name)