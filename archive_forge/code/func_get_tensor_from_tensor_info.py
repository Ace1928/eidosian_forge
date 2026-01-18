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
@tf_export(v1=['saved_model.get_tensor_from_tensor_info', 'saved_model.utils.get_tensor_from_tensor_info'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def get_tensor_from_tensor_info(tensor_info, graph=None, import_scope=None):
    """Returns the Tensor or CompositeTensor described by a TensorInfo proto.

  Args:
    tensor_info: A TensorInfo proto describing a Tensor or SparseTensor or
      CompositeTensor.
    graph: The tf.Graph in which tensors are looked up. If None, the
        current default graph is used.
    import_scope: If not None, names in `tensor_info` are prefixed with this
        string before lookup.

  Returns:
    The Tensor or SparseTensor or CompositeTensor in `graph` described by
    `tensor_info`.

  Raises:
    KeyError: If `tensor_info` does not correspond to a tensor in `graph`.
    ValueError: If `tensor_info` is malformed.
  """
    graph = graph or ops.get_default_graph()

    def _get_tensor(name):
        return graph.get_tensor_by_name(ops.prepend_name_scope(name, import_scope=import_scope))
    encoding = tensor_info.WhichOneof('encoding')
    if encoding == 'name':
        return _get_tensor(tensor_info.name)
    elif encoding == 'coo_sparse':
        return sparse_tensor.SparseTensor(_get_tensor(tensor_info.coo_sparse.indices_tensor_name), _get_tensor(tensor_info.coo_sparse.values_tensor_name), _get_tensor(tensor_info.coo_sparse.dense_shape_tensor_name))
    elif encoding == 'composite_tensor':
        spec_proto = struct_pb2.StructuredValue(type_spec_value=tensor_info.composite_tensor.type_spec)
        spec = nested_structure_coder.decode_proto(spec_proto)
        components = [_get_tensor(component.name) for component in tensor_info.composite_tensor.components]
        return nest.pack_sequence_as(spec, components, expand_composites=True)
    else:
        raise ValueError(f'Invalid TensorInfo.encoding: {encoding}. Expected `coo_sparse`, `composite_tensor`, or `name` for a dense tensor.')