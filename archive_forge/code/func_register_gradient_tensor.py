import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def register_gradient_tensor(self, x_tensor_name, gradient_tensor):
    """Register the gradient tensor for an x-tensor.

    Args:
      x_tensor_name: (`str`) the name of the independent `tf.Tensor`, i.e.,
        the tensor on the denominator of the differentiation.
      gradient_tensor: the gradient `tf.Tensor`.
    """
    if len(_gradient_debuggers) == 1 or self._is_active_context:
        self._check_same_graph(gradient_tensor)
        self._gradient_tensors[x_tensor_name] = gradient_tensor