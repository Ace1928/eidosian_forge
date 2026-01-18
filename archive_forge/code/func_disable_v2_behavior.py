from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.experimental.ops import readers as exp_readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['disable_v2_behavior'])
def disable_v2_behavior():
    """Disables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 1.x.

  User can call this function to disable 2.x behavior during complex migrations.

  @compatibility(TF2)
  Using this function indicates that your software is not compatible
  with eager execution and `tf.function` in TF2.

  To migrate to TF2, rewrite your code to be compatible with eager execution.
  Please refer to the [migration guide]
  (https://www.tensorflow.org/guide/migrate) for additional resource on the
  topic.
  @end_compatibility
  """
    _v2_behavior_usage_gauge.get_cell('disable').set(True)
    tf2.disable()
    ops.disable_eager_execution()
    tensor_shape.disable_v2_tensorshape()
    variable_scope.disable_resource_variables()
    tensor.disable_tensor_equality()
    control_flow_v2_toggles.disable_control_flow_v2()
    dataset_ops.Dataset = dataset_ops.DatasetV1
    readers.FixedLengthRecordDataset = readers.FixedLengthRecordDatasetV1
    readers.TFRecordDataset = readers.TFRecordDatasetV1
    readers.TextLineDataset = readers.TextLineDatasetV1
    counter.Counter = counter.CounterV1
    interleave_ops.choose_from_datasets = interleave_ops.choose_from_datasets_v1
    interleave_ops.sample_from_datasets = interleave_ops.sample_from_datasets_v1
    random_ops.RandomDataset = random_ops.RandomDatasetV1
    exp_readers.CsvDataset = exp_readers.CsvDatasetV1
    exp_readers.SqlDataset = exp_readers.SqlDatasetV1
    exp_readers.make_batched_features_dataset = exp_readers.make_batched_features_dataset_v1
    exp_readers.make_csv_dataset = exp_readers.make_csv_dataset_v1