from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops
def sleep(sleep_microseconds):
    """Sleeps for `sleep_microseconds` before producing each input element.

  Args:
    sleep_microseconds: The number of microseconds to sleep before producing an
      input element.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

    def _apply_fn(dataset):
        return _SleepDataset(dataset, sleep_microseconds)
    return _apply_fn