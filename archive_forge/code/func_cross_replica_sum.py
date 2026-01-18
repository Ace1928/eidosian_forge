from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.cross_replica_sum'])
def cross_replica_sum(x, group_assignment=None, name=None):
    """Sum the input tensor across replicas according to group_assignment.

  Args:
    x: The local tensor to the sum.
    group_assignment: Optional 2d int32 lists with shape [num_groups,
      num_replicas_per_group]. `group_assignment[i]` represents the replica ids
      in the ith subgroup.
    name: Optional op name.

  Returns:
    A `Tensor` which is summed across replicas.
  """
    if group_assignment is None:
        group_assignment = _create_default_group_assignment()
    return gen_tpu_ops.cross_replica_sum(x, group_assignment, name=name)