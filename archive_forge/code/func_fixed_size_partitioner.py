import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['fixed_size_partitioner'])
def fixed_size_partitioner(num_shards, axis=0):
    """Partitioner to specify a fixed number of shards along given axis.

  @compatibility(TF2)
  This API is deprecated in TF2. In TF2, partitioner is no longer part of
  the variable declaration via `tf.Variable`.
  [ParameterServer Training]
  (https://www.tensorflow.org/tutorials/distribute/parameter_server_training)
  handles partitioning of variables. The corresponding TF2 partitioner class of
  `fixed_size_partitioner` is
  `tf.distribute.experimental.partitioners.FixedShardsPartitioner`.

  Check the [migration guide]
  (https://www.tensorflow.org/guide/migrate#2_use_python_objects_to_track_variables_and_losses)
  on the differences in treatment of variables and losses between TF1 and TF2.

  Before:

    ```
    x = tf.compat.v1.get_variable(
      "x", shape=(2,), partitioner=tf.compat.v1.fixed_size_partitioner(2)
    )
    ```
  After:

    ```
    partitioner = (
        tf.distribute.experimental.partitioners.FixedShardsPartitioner(
            num_shards=2)
    )
    strategy = tf.distribute.experimental.ParameterServerStrategy(
                   cluster_resolver=cluster_resolver,
                   variable_partitioner=partitioner)

    with strategy.scope():
      x = tf.Variable([1.0, 2.0])
    ```
  @end_compatibility

  Args:
    num_shards: `int`, number of shards to partition variable.
    axis: `int`, axis to partition on.

  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope` and `get_variable`.
  """

    def _partitioner(shape, **unused_args):
        partitions_list = [1] * len(shape)
        partitions_list[axis] = min(num_shards, shape.dims[axis].value)
        return partitions_list
    return _partitioner