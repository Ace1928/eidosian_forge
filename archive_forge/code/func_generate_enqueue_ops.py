import itertools
import numpy as np
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_sharding
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
def generate_enqueue_ops(self, sharded_inputs):
    """Generates the host-side Ops to enqueue the partitioned inputs.

    sharded_inputs is a list, one for each replica, of lists of
    Tensors. sharded_inputs[i] is the tuple of Tensors to use to feed
    replica i.
    sharded_inputs[i][j] is partitioned by self._input_partition_dims[j].

    For example, if sharded_inputs[i][j] is a 2-D Tensor:
    [[A, B, C, D],
     [E ,F, G, H]]
    self._input_partition_dims[j] is [2, 4].

    sharded_inputs[i][j] will be partitioned and flattened into:
    [A, B, C, D, E, F, G, H] and fed into the logical core ids:
    [0, 1, 2, 3, 4, 5, 6, 7] respectively.

    Args:
      sharded_inputs: a list of lists of Tensors. The length of the
        outer list determines the number of shards. Each inner list indicates
        the types and shapes of the tuples in the corresponding shard.

    Returns:
      A list of host-side Ops, one for each shard, that when executed together
      will enqueue a full-size element of infeed.

    Raises:
      ValueError: if the queue configuration has previously been frozen and the
        shapes of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the shapes of the elements of sharded_inputs
        don't form a consistent unsharded tuple; or if the elements of a tuple
        have different device constraints; or if the partition dims are invalid.
      TypeError: if the queue configuration has previously been frozen and the
        types of the elements of sharded_inputs are not compatible with the
        frozen configuration; or if the types of the elements of sharded_inputs
        don't form a consistent unsharded tuple.
    """
    self.set_configuration_from_sharded_input_tensors(sharded_inputs)
    number_of_replicas = len(sharded_inputs)
    number_of_tuple_elements = len(sharded_inputs[0])
    assert len(self._input_partition_dims) == number_of_tuple_elements
    enqueue_ops = []
    for replica_index in range(number_of_replicas):
        flattened_inputs = sharded_inputs[replica_index]
        inputs_part_dims_flat = nest.flatten_up_to(flattened_inputs, self._input_partition_dims)
        inputs_parted_iters = [iter(self._check_dims_and_partition_or_replicate_on_host(x, dims)) for x, dims in zip(sharded_inputs[replica_index], inputs_part_dims_flat)]
        replica_id = self._device_assignment.lookup_replicas(task_id=self._host_id, logical_core=0)[replica_index]
        for logical_core in range(self._device_assignment.num_cores_per_replica):
            device = self._device_assignment.host_device(replica=replica_id, logical_core=logical_core)
            with ops.device(device):
                ordinal = self._device_assignment.tpu_ordinal(replica=replica_id, logical_core=logical_core)
                infeed_inputs = []
                for it in inputs_parted_iters:
                    input_for_device = next(it, None)
                    if input_for_device is not None:
                        infeed_inputs.append(input_for_device)
                if infeed_inputs:
                    enqueue_ops.append(tpu_ops.infeed_enqueue_tuple(inputs=infeed_inputs, shapes=[x.shape for x in infeed_inputs], name='enqueue/replica_{0}/input_{1}'.format(replica_index, logical_core), device_ordinal=ordinal))
    return enqueue_ops