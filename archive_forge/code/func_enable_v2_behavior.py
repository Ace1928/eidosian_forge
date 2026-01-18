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
@tf_export(v1=['enable_v2_behavior'])
def enable_v2_behavior():
    """Enables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 2.x.

  This function is called in the main TensorFlow `__init__.py` file, user should
  not need to call it, except during complex migrations.

  @compatibility(TF2)
  This function is not necessary if you are using TF2. V2 behavior is enabled by
  default.
  @end_compatibility
  """
    _v2_behavior_usage_gauge.get_cell('enable').set(True)
    tf2.enable()
    ops.enable_eager_execution()
    tensor_shape.enable_v2_tensorshape()
    variable_scope.enable_resource_variables()
    tensor.enable_tensor_equality()
    control_flow_v2_toggles.enable_control_flow_v2()
    dataset_ops.Dataset = dataset_ops.DatasetV2
    readers.FixedLengthRecordDataset = readers.FixedLengthRecordDatasetV2
    readers.TFRecordDataset = readers.TFRecordDatasetV2
    readers.TextLineDataset = readers.TextLineDatasetV2
    counter.Counter = counter.CounterV2
    interleave_ops.choose_from_datasets = interleave_ops.choose_from_datasets_v2
    interleave_ops.sample_from_datasets = interleave_ops.sample_from_datasets_v2
    random_ops.RandomDataset = random_ops.RandomDatasetV2
    exp_readers.CsvDataset = exp_readers.CsvDatasetV2
    exp_readers.SqlDataset = exp_readers.SqlDatasetV2
    exp_readers.make_batched_features_dataset = exp_readers.make_batched_features_dataset_v2
    exp_readers.make_csv_dataset = exp_readers.make_csv_dataset_v2