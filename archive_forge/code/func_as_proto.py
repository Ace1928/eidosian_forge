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
def as_proto(self):
    """Returns this shape as a `TensorShapeProto`."""
    if self._dims is None:
        return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
        return tensor_shape_pb2.TensorShapeProto(dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=-1 if d is None else d) for d in self._dims])