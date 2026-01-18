from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def is_debug_node(node_name):
    """Determine whether a node name is that of a debug node.

  Such nodes are inserted by TensorFlow core upon request in
  RunOptions.debug_options.debug_tensor_watch_opts.

  Args:
    node_name: Name of the node.

  Returns:
    A bool indicating whether the input argument is the name of a debug node.
  """
    return node_name.startswith('__dbg_')