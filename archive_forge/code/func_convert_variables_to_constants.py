import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions='This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.')
@tf_export(v1=['graph_util.convert_variables_to_constants'])
def convert_variables_to_constants(sess, input_graph_def, output_node_names, variable_names_whitelist=None, variable_names_blacklist=None):
    """Replaces all the variables in a graph with constants of the same values.

  If you have a trained graph containing Variable ops, it can be convenient to
  convert them all to Const ops holding the same values. This makes it possible
  to describe the network fully with a single GraphDef file, and allows the
  removal of a lot of ops related to loading and saving the variables.

  Args:
    sess: Active TensorFlow session containing the variables.
    input_graph_def: GraphDef object holding the network.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_whitelist: The set of variable names to convert (by default,
      all variables are converted).
    variable_names_blacklist: The set of variable names to omit converting to
      constants.

  Returns:
    GraphDef containing a simplified version of the original.

  Raises:
    RuntimeError: if a DT_RESOURCE op is found whose ancestor Variables are both
      denylisted AND whitelisted for freezing.
  """
    ret = convert_variables_to_constants_from_session_graph(session=sess, graph_def=input_graph_def, output_node_names=output_node_names, variable_names_allowlist=variable_names_whitelist, variable_names_denylist=variable_names_blacklist)
    return ret