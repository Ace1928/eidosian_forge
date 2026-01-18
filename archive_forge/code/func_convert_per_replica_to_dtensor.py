from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
def convert_per_replica_to_dtensor(per_replica_value, mesh):
    """Convert a PerReplica result to a DTensor instance.

  Args:
    per_replica_value: A PerReplica instance whose value will be converted
      to DTensor.
    mesh: The mesh used for layout creation.

  Returns:
    A DTensor instance that packed from per_replica_value with batch sharded
      layout.
  """
    values = per_replica_value.values
    if isinstance(values[0], (float, int)):
        rank = 0
    else:
        rank = len(values[0].shape)
    if rank == 0:
        result = []
        for v in values:
            result.append(array_ops.expand_dims_v2(v, axis=0))
        rank += 1
    else:
        result = list(values)
    batch_layout = layout.Layout.batch_sharded(mesh, batch_dim=DEFAULT_BATCH_MESH_DIM_NAME, rank=rank)
    return d_api.pack(result, batch_layout)