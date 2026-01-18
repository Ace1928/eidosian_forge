from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def build_tensor_info_internal(tensor):
    """Utility function to build TensorInfo proto from a Tensor."""
    if isinstance(tensor, composite_tensor.CompositeTensor) and (not isinstance(tensor, sparse_tensor.SparseTensor)) and (not isinstance(tensor, resource_variable_ops.ResourceVariable)):
        return _build_composite_tensor_info_internal(tensor)
    tensor_info = meta_graph_pb2.TensorInfo(dtype=dtypes.as_dtype(tensor.dtype).as_datatype_enum, tensor_shape=tensor.get_shape().as_proto())
    if isinstance(tensor, sparse_tensor.SparseTensor):
        tensor_info.coo_sparse.values_tensor_name = tensor.values.name
        tensor_info.coo_sparse.indices_tensor_name = tensor.indices.name
        tensor_info.coo_sparse.dense_shape_tensor_name = tensor.dense_shape.name
    else:
        tensor_info.name = tensor.name
    return tensor_info