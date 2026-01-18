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
def assert_is_fully_defined(self):
    """Raises an exception if `self` is not fully defined in every dimension.

    Raises:
      ValueError: If `self` does not have a known value for every dimension.
    """
    if not self.is_fully_defined():
        raise ValueError('Shape %s is not fully defined' % self)