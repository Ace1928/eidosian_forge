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
def convert_variables_to_constants_v2_as_graph(func, lower_control_flow=True, aggressive_inlining=False):
    """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  returns the intermediate `GraphDef` as well. This `GraphDef` contains all the
  debug information after all the transformations in the frozen phase.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs).

  Returns:
    ConcreteFunction containing a simplified version of the original, and also
    the intermediate GraphDef containing the node debug information for the
    transformations in the frozen phase.
  """
    converter_data = _FunctionConverterDataInEager(func=func, lower_control_flow=lower_control_flow, aggressive_inlining=aggressive_inlining)
    output_graph_def, converted_input_indices = _replace_variables_by_constants(converter_data=converter_data)
    frozen_func = _construct_concrete_function(func, output_graph_def, converted_input_indices)
    return (frozen_func, output_graph_def)