import collections
import math
import re
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib
def optimize_for_inference(input_graph_def, input_node_names, output_node_names, placeholder_type_enum, toco_compatible=False):
    """Applies a series of inference optimizations on the input graph.

  Args:
    input_graph_def: A GraphDef containing a training model.
    input_node_names: A list of names of the nodes that are fed inputs during
      inference.
    output_node_names: A list of names of the nodes that produce the final
      results.
    placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.
    toco_compatible: Boolean, if True, only runs optimizations that result in
      TOCO compatible graph operations (default=False).

  Returns:
    An optimized version of the input graph.
  """
    ensure_graph_is_valid(input_graph_def)
    optimized_graph_def = input_graph_def
    optimized_graph_def = strip_unused_lib.strip_unused(optimized_graph_def, input_node_names, output_node_names, placeholder_type_enum)
    optimized_graph_def = graph_util.remove_training_nodes(optimized_graph_def, output_node_names)
    optimized_graph_def = fold_batch_norms(optimized_graph_def)
    if not toco_compatible:
        optimized_graph_def = fuse_resize_and_conv(optimized_graph_def, output_node_names)
    ensure_graph_is_valid(optimized_graph_def)
    return optimized_graph_def