from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.experimental.prefetch_to_device')
def prefetch_to_device(device, buffer_size=None):
    """A transformation that prefetches dataset values to the given `device`.

  NOTE: Although the transformation creates a `tf.data.Dataset`, the
  transformation must be the final `Dataset` in the input pipeline.

  For example,
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
  >>> for element in dataset:
  ...   print(f'Tensor {element} is on device {element.device}')
  Tensor 1 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 2 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 3 is on device /job:localhost/replica:0/task:0/device:CPU:0

  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

    def _apply_fn(dataset):
        return dataset.apply(copy_to_device(target_device=device)).prefetch(buffer_size)
    return _apply_fn