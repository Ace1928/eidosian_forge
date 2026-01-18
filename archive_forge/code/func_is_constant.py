import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def is_constant(tensor_or_op):
    if isinstance(tensor_or_op, tensor_lib.Tensor):
        op = tensor_or_op.op
    else:
        op = tensor_or_op
    return op.type == 'Const'