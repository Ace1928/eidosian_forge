import collections
import functools
import itertools
import wrapt
from tensorflow.python.data.util import nest
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import internal
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.nest_util import CustomNestProtocol
from tensorflow.python.util.tf_export import tf_export
def convert_legacy_structure(output_types, output_shapes, output_classes):
    """Returns a `Structure` that represents the given legacy structure.

  This method provides a way to convert from the existing `Dataset` and
  `Iterator` structure-related properties to a `Structure` object. A "legacy"
  structure is represented by the `tf.data.Dataset.output_types`,
  `tf.data.Dataset.output_shapes`, and `tf.data.Dataset.output_classes`
  properties.

  TODO(b/110122868): Remove this function once `Structure` is used throughout
  `tf.data`.

  Args:
    output_types: A nested structure of `tf.DType` objects corresponding to
      each component of a structured value.
    output_shapes: A nested structure of `tf.TensorShape` objects
      corresponding to each component a structured value.
    output_classes: A nested structure of Python `type` objects corresponding
      to each component of a structured value.

  Returns:
    A `Structure`.

  Raises:
    TypeError: If a structure cannot be built from the arguments, because one of
      the component classes in `output_classes` is not supported.
  """
    flat_types = nest.flatten(output_types)
    flat_shapes = nest.flatten(output_shapes)
    flat_classes = nest.flatten(output_classes)
    flat_ret = []
    for flat_type, flat_shape, flat_class in zip(flat_types, flat_shapes, flat_classes):
        if isinstance(flat_class, type_spec.TypeSpec):
            flat_ret.append(flat_class)
        elif issubclass(flat_class, sparse_tensor.SparseTensor):
            flat_ret.append(sparse_tensor.SparseTensorSpec(flat_shape, flat_type))
        elif issubclass(flat_class, tensor_lib.Tensor):
            flat_ret.append(tensor_lib.TensorSpec(flat_shape, flat_type))
        elif issubclass(flat_class, tensor_array_ops.TensorArray):
            flat_ret.append(tensor_array_ops.TensorArraySpec(flat_shape[2:], flat_type, dynamic_size=tensor_shape.dimension_value(flat_shape[0]), infer_shape=tensor_shape.dimension_value(flat_shape[1])))
        else:
            raise TypeError('Could not build a structure for output class {}. Make sure any component class in `output_classes` inherits from one of the following classes: `tf.TypeSpec`, `tf.sparse.SparseTensor`, `tf.Tensor`, `tf.TensorArray`.'.format(flat_class.__name__))
    return nest.pack_sequence_as(output_classes, flat_ret)