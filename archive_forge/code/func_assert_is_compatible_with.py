import functools
import operator
from typing import Optional, Sequence, Type
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def assert_is_compatible_with(self, other):
    """Raises exception if `self` and `other` do not represent the same shape.

    This method can be used to assert that there exists a shape that both
    `self` and `other` represent.

    Args:
      other: Another TensorShape.

    Raises:
      ValueError: If `self` and `other` do not represent the same shape.
    """
    if not self.is_compatible_with(other):
        raise ValueError('Shapes %s and %s are incompatible' % (self, other))