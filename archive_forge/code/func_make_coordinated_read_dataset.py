import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def make_coordinated_read_dataset(self, cluster, num_consumers, sharding_policy=data_service_ops.ShardingPolicy.OFF):
    """Creates a dataset that performs coordinated reads.

    The dataset simulates `num_consumers` consumers by using parallel
    interleave to read with `num_consumers` threads, one for each consumer. The
    nth element of the dataset is produced by consumer `n % num_consumers`.

    The dataset executed on each worker will produce groups of `num_consumers`
    sequentially increasing numbers. For example, if `num_consumers=3` a worker
    dataset could produce [0, 1, 2, 9, 10, 11, 21, 22, 23]. This enables
    `checkCoordinatedReadGroups` below to assess whether the values received in
    each step came from the same group.

    Args:
      cluster: A tf.data service `TestCluster`.
      num_consumers: The number of consumers to simulate.
      sharding_policy: The sharding policy to use. Currently only OFF and
        DYNAMIC are supported.

    Returns:
      A dataset that simulates reading with `num_consumers` consumers.
    """
    if sharding_policy not in [data_service_ops.ShardingPolicy.OFF, data_service_ops.ShardingPolicy.DYNAMIC]:
        raise ValueError(f'Unsupported sharding policy: {sharding_policy}')
    ds = dataset_ops.Dataset.from_tensors(math_ops.cast(0, dtypes.int64))
    ds = ds.concatenate(dataset_ops.Dataset.random())

    def make_group(x):
        x = x % 2 ** 32
        return dataset_ops.Dataset.range(x * num_consumers, (x + 1) * num_consumers)
    ds = ds.flat_map(make_group)
    consumers = []
    for consumer_index in range(num_consumers):
        consumers.append(self.make_distributed_dataset(ds, cluster, job_name='test', processing_mode=sharding_policy, consumer_index=consumer_index, num_consumers=num_consumers))
    ds = dataset_ops.Dataset.from_tensor_slices(consumers)
    ds = ds.interleave(lambda x: x, cycle_length=num_consumers, num_parallel_calls=num_consumers)
    return ds