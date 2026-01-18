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
def num_elements(self):
    """Returns the total number of elements, or none for incomplete shapes."""
    if self.is_fully_defined():
        return functools.reduce(operator.mul, self.as_list(), 1)
    else:
        return None