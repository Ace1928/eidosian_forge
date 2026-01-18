import warnings
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource
def get_tensor_from_node(node):
    """Resolves a saved model graph node into a tensor to be captured.

  Args:
    node: a tensor, variable, or resource to be resolved into a capturable
      tensor

  Returns:
    A list of tensors.
  Raises:
    ValueError: if the node cannot be converted into a tensor.
  """
    with ops.init_scope():
        if getattr(node, 'is_distributed_variable', False):
            return node
        elif getattr(node, 'is_distributed_table', False):
            return node
        elif getattr(node, 'is_sharded_variable', False):
            return node
        elif resource_variable_ops.is_resource_variable(node):
            return node.handle
        elif isinstance(node, asset.Asset):
            return node.asset_path
        elif tensor_util.is_tf_type(node):
            return node
        elif isinstance(node, resource.CapturableResource):
            return node.resource_handle
        raise ValueError(f'Cannot convert node {node} to tensor.')