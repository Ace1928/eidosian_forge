from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.assert_global_step'])
def assert_global_step(global_step_tensor):
    """Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  """
    if not (isinstance(global_step_tensor, variables.Variable) or isinstance(global_step_tensor, tensor.Tensor) or resource_variable_ops.is_resource_variable(global_step_tensor)):
        raise TypeError('Existing "global_step" must be a Variable or Tensor: %s.' % global_step_tensor)
    if not global_step_tensor.dtype.base_dtype.is_integer:
        raise TypeError('Existing "global_step" does not have integer type: %s' % global_step_tensor.dtype)
    if global_step_tensor.get_shape().ndims != 0 and global_step_tensor.get_shape().is_fully_defined():
        raise TypeError('Existing "global_step" is not scalar: %s' % global_step_tensor.get_shape())