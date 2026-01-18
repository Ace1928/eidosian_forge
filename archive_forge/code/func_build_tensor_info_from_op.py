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
def build_tensor_info_from_op(op):
    """Utility function to build TensorInfo proto from an Op.

  Note that this function should be used with caution. It is strictly restricted
  to TensorFlow internal use-cases only. Please make sure you do need it before
  using it.

  This utility function overloads the TensorInfo proto by setting the name to
  the Op's name, dtype to DT_INVALID and tensor_shape as None. One typical usage
  is for the Op of the call site for the defunned function:
  ```python
    @function.defun
    def some_variable_initialization_fn(value_a, value_b):
      a = value_a
      b = value_b

    value_a = constant_op.constant(1, name="a")
    value_b = constant_op.constant(2, name="b")
    op_info = utils.build_op_info(
        some_variable_initialization_fn(value_a, value_b))
  ```

  Args:
    op: An Op whose name is used to build the TensorInfo. The name that points
        to the Op could be fetched at run time in the Loader session.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.

  Raises:
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError('`build_tensor_info_from_op` is not supported in eager execution.')
    return meta_graph_pb2.TensorInfo(dtype=types_pb2.DT_INVALID, tensor_shape=tensor_shape.unknown_shape().as_proto(), name=op.name)