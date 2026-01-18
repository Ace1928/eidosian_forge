from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.data.Dataset.sample_from_datasets(...)`.')
@tf_export(v1=['data.experimental.sample_from_datasets'])
def sample_from_datasets_v1(datasets, weights=None, seed=None, stop_on_empty_dataset=False):
    return dataset_ops.DatasetV1Adapter(sample_from_datasets_v2(datasets, weights, seed, stop_on_empty_dataset))