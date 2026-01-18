from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
def keras_tensor_from_type_spec(type_spec, name=None):
    """Convert a TypeSpec to a representative KerasTensor."""
    keras_tensor_cls = None
    value_type = type_spec.value_type
    for tensor_type, cls in keras_tensor_classes:
        if issubclass(value_type, tensor_type):
            keras_tensor_cls = cls
            break
    return keras_tensor_cls.from_type_spec(type_spec, name=name)