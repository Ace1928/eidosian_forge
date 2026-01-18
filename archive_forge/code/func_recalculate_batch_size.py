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
def recalculate_batch_size(type_spec):
    """Recalculates the output_shape after dividing it by num_replicas."""
    output_shape = type_spec._to_legacy_output_shapes()
    if not isinstance(output_shape, tensor_shape.TensorShape):
        return None
    if output_shape.rank is None:
        return None
    if len(output_shape) < 1:
        raise ValueError('Invalid `input_dataset`. Expected a dataset whose elements have rank >= 1 but found a dataset whose elements are scalars. Fix the issue by adding the `batch` transformation to the dataset.')
    output_dims = [d.value for d in output_shape.dims]
    if output_dims[0] is not None and output_dims[0] % num_replicas == 0:
        return output_dims[0] // num_replicas
    return None