from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
def assert_next(transformations):
    """A transformation that asserts which transformations happen next.

  Transformations should be referred to by their base name, not including
  version suffix. For example, use "Batch" instead of "BatchV2". "Batch" will
  match any of "Batch", "BatchV1", "BatchV2", etc.

  Args:
    transformations: A `tf.string` vector `tf.Tensor` identifying the
      transformations that are expected to happen next.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

    def _apply_fn(dataset):
        """Function from `Dataset` to `Dataset` that applies the transformation."""
        return _AssertNextDataset(dataset, transformations)
    return _apply_fn