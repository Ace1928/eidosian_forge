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
@tf_export('experimental.dtensor.name_based_save', v1=[])
def name_based_save(mesh: layout_lib.Mesh, checkpoint_prefix: Union[str, tensor_lib.Tensor], name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]):
    """Saves name based Tensor into a Checkpoint.

  The function prepares the input dictionary to the format of a `sharded_save`,
  so that it can take advantage of DTensor SPMD based distributed save.

  Same as restore, the function only supports saving on the single mesh.

  Args:
    mesh: The single mesh that all Tensors would be restored to.
    checkpoint_prefix : The prefix of checkpoint to be restored.
    name_tensor_dict: A ordered dictionary of tensor_names to a DTensor. The
      DTensor shape/dtype must match the tensors being saved/restored for now.
  """
    if not context.executing_eagerly():
        raise ValueError('name based save must run eagerly.')
    ordered_name_tensor_dict = name_tensor_dict
    if not isinstance(name_tensor_dict, collections.OrderedDict):
        ordered_name_tensor_dict = collections.OrderedDict(name_tensor_dict)
    checkpoint_prefix = api.pack([checkpoint_prefix] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=0))
    tensor_names = api.pack([list(ordered_name_tensor_dict.keys())] * mesh.num_local_devices(), layout_lib.Layout.replicated(mesh.host_mesh(), rank=1))
    sharded_save(mesh, file_prefix=checkpoint_prefix, tensor_names=tensor_names, shape_and_slices=[''] * len(ordered_name_tensor_dict), tensors=list(ordered_name_tensor_dict.values()))