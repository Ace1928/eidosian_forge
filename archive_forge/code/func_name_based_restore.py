import collections
from typing import Dict, List, Union
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.name_based_restore', v1=[])
def name_based_restore(mesh: layout_lib.Mesh, checkpoint_prefix: str, name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]):
    """Restores from checkpoint_prefix to name based DTensors.

  It is required to have already-initialized DTensor variables that have same
  shape/dtype for the tensors being restored.

  Also, we currently only support a named based restore on a single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.

  Returns:
    A dictionary of name to its restored DTensor value.
  """
    if not context.executing_eagerly():
        raise ValueError('name based restore must run eagerly.')
    ordered_name_tensor_dict = name_tensor_dict
    if not isinstance(name_tensor_dict, collections.OrderedDict):
        ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)
    for name, tensor in ordered_name_tensor_dict.items():
        try:
            if api.fetch_layout(tensor).mesh.device_type().upper() != 'CPU':
                raise ValueError('Restoring a non CPU Tensor is not supported currently. Offending tensor name : {tensor_name}'.format(tensor_name=name))
        except errors_impl.OpError as op_error:
            raise ValueError('Saving/Restoring tensor must be a DTensor') from op_error
    checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
    tensor_names = api.pack([list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    shape_and_slices = api.pack([[''] * len(ordered_name_tensor_dict)] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    input_shapes = [tensor.shape for tensor in ordered_name_tensor_dict.values()]
    input_layouts = [api.fetch_layout(tensor).to_string() for tensor in ordered_name_tensor_dict.values()]
    with ops.device(api.device_name()):
        restored_cpu_tensors = gen_dtensor_ops.d_tensor_restore_v2(prefix=checkpoint_prefix, tensor_names=tensor_names, shape_and_slices=shape_and_slices, input_shapes=input_shapes, input_layouts=input_layouts, dtypes=[tensor.dtype for tensor in ordered_name_tensor_dict.values()])
    return collections.OrderedDict(zip(ordered_name_tensor_dict.keys(), restored_cpu_tensors))