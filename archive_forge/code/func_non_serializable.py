from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
def non_serializable():
    """A non-serializable identity transformation.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

    def _apply_fn(dataset):
        """Function from `Dataset` to `Dataset` that applies the transformation."""
        return _NonSerializableDataset(dataset)
    return _apply_fn