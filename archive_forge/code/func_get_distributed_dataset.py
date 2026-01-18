from tensorflow.python import tf2
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
def get_distributed_dataset(dataset, input_workers, strategy, num_replicas_in_sync=None, input_context=None, options=None, build=True, replica_order=None):
    """Returns a distributed dataset from the given tf.data.Dataset instance.

  This is a common function that is used by all strategies to return a
  distributed dataset. The distributed dataset instance returned is different
  depending on if we are in a TF 1 or TF 2 context. The distributed dataset
  instances returned differ from each other in the APIs supported by each of
  them.

  Args:
    dataset: a tf.data.Dataset instance.
    input_workers: an InputWorkers object which specifies devices on which
      iterators should be created.
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
      handle last partial batch.
    num_replicas_in_sync: Optional integer. If this is not None, the value is
      used to decide how to rebatch datasets into smaller batches so that the
      total batch size for each step (across all workers and replicas) adds up
      to `dataset`'s batch size.
    input_context: `InputContext` for sharding. Only pass this in for between
      graph multi-worker cases where there is only one `input_worker`. In these
      cases, we will shard based on the `input_pipeline_id` and
      `num_input_pipelines` in the `InputContext`.
    options: Default is None. `tf.distribute.InputOptions` used to control
      options on how this dataset is distributed.
    build: whether to build underlying datasets when a DistributedDataset is
      created. This is only useful for `ParameterServerStrategy` now.
    replica_order: the order of the replicas, which will be used to reorder the
      iterators to match the device order.

  Returns:
    A distributed dataset instance.
  """
    if tf2.enabled():
        return input_lib.DistributedDataset(input_workers, strategy, dataset, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context, build=build, options=options, replica_order=replica_order)
    else:
        return input_lib_v1.DistributedDatasetV1(dataset, input_workers, strategy, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context, options=options)