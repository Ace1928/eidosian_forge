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
def convert_variables_to_constants_from_session_graph(session, graph_def, output_node_names, variable_names_allowlist=None, variable_names_denylist=None):
    """Replaces all the variables in a graph with constants of the same values.

  This function works similarly to convert_variables_to_constants_v2, but it
  retrieves the constant values from a Session instead of from a
  ConcreteFunction. This is useful when converting graphs generated from
  TensorFlow V1, where ConcreteFunctions are not available. This also differs
  from graph_util.convert_variables_to_constants in that it supports resource
  variables when V2 control flow constructions are present.

  Args:
    session: Active TensorFlow session containing the variables.
    graph_def: A GraphDef to convert.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_allowlist: The set of variable names to convert (by default,
      all variables are converted).
    variable_names_denylist: The set of variable names to omit converting to
      constants.

  Returns:
    An optimized GraphDef.
  """
    graph_def, _ = _replace_variables_by_constants(converter_data=_SessionConverterData(session=session, graph_def=graph_def, output_node_names=output_node_names, variable_names_allowlist=variable_names_allowlist, variable_names_denylist=variable_names_denylist))
    return graph_def