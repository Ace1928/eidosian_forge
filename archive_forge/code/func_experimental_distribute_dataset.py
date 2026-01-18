from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export
def experimental_distribute_dataset(self, dataset, options=None):
    """Distributes a tf.data.Dataset instance provided via dataset.

    The returned dataset is a wrapped strategy dataset which creates a
    multidevice iterator under the hood. It prefetches the input data to the
    specified devices on the worker. The returned distributed dataset can be
    iterated over similar to how regular datasets can.

    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    For Example:
    ```
    strategy = tf.distribute.CentralStorageStrategy()  # with 1 CPU and 1 GPU
    dataset = tf.data.Dataset.range(10).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    for x in dist_dataset:
      print(x)  # Prints PerReplica values [0, 1], [2, 3],...

    ```
    Args:
      dataset: `tf.data.Dataset` to be prefetched to device.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.

    Returns:
      A "distributed `Dataset`" that the caller can iterate over.
    """
    if options and options.experimental_replication_moden == distribute_lib.InputReplicationMode.PER_REPLICA:
        raise NotImplementedError('InputReplicationMode.PER_REPLICA is only supported in `experimental_distribute_datasets_from_function`.')
    return super(CentralStorageStrategy, self).experimental_distribute_dataset(dataset, options)