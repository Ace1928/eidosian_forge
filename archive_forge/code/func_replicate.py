from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import tf_export
def replicate(dataset, devices):
    """A transformation that replicates `dataset` onto a list of devices.

  Args:
    dataset: A `tf.data.Dataset` object.
    devices: A list of devices to replicate the dataset on.

  Returns:
    A dictionary mapping device name to a dataset on that device.
  """
    if not isinstance(dataset, data_types.DatasetV2):
        raise TypeError(f'Invalid `dataset`. Expected a `tf.data.Dataset` object but got {type(dataset)}.')
    dataset_device = dataset._variant_tensor.device
    datasets = {}
    if len(devices) == 1 and devices[0] == dataset_device:
        datasets[devices[0]] = dataset
        return datasets
    with ops.colocate_with(dataset._variant_tensor):
        dataset = dataset._apply_debug_options()
        graph_def = dataset._as_serialized_graph(strip_device_assignment=True, external_state_policy=ExternalStatePolicy.WARN)
    for device in devices:
        ds = _RemoteDataset(graph_def, device, dataset.element_spec)
        datasets[device] = ds
    return datasets