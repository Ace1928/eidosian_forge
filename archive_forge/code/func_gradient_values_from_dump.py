import re
import uuid
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
def gradient_values_from_dump(grad_debugger, x_tensor, dump):
    """Find gradient values from a `DebugDumpDir` object.

  Args:
    grad_debugger: the `tf_debug.GradientsDebugger` instance to be used.
    x_tensor: (`tf.Tensor`, `tf.Variable` or `str`) The x-tensor object or its
      name. x-tensor refers to the independent `tf.Tensor`, i.e., the tensor
      on the denominator of the differentiation.
    dump: A `tfdbg.DebugDumpDir` object.

  Returns:
    If this `GradientsDebugger` instance has the gradient tensor of `x_tensor`
      registered: a list of `numpy.ndarray` representing the value of the
      gradient tensor from `dump`. The list could be empty, if the gradient
      tensor is not executed in the `tf.Session.run()` call that generated
      the `dump`. The list could also contain multiple values of the gradient
      tensor, e.g., if gradient tensor is computed repeatedly in a
      `tf.while_loop` during the run that generated the `dump`.

  Raises:
    LookupError: If this `GradientsDebugger` instance does not have the
      gradient tensor of `x_tensor` registered.
    ValueError: If this `GradientsDebugger` has a `tf.Graph` object that
      does not match the `tf.Graph` object of the `dump`.
    TypeError: If `x_tensor` is not a `tf.Tensor`, `tf.Variable` or `str`.
  """
    if dump.python_graph and grad_debugger.graph and (dump.python_graph != grad_debugger.graph):
        raise ValueError('This GradientsDebugger instance has a graph (%s) that differs from the graph of the DebugDumpDir object (%s).' % (grad_debugger.graph, dump.python_graph))
    gradient_tensor = grad_debugger.gradient_tensor(x_tensor)
    node_name, output_slot = debug_graphs.parse_node_or_tensor_name(gradient_tensor.name)
    try:
        return dump.get_tensors(node_name, output_slot, 'DebugIdentity')
    except debug_data.WatchKeyDoesNotExistInDebugDumpDirError:
        return []