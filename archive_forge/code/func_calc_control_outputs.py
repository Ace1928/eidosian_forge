import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
def calc_control_outputs(self, graph):
    """Returns the map of control_outputs for a given graph.

    Args:
      graph: The graph to parse.

    Returns:
      A map of the control outputs.
    """
    control_outputs = {}
    for op in graph.get_operations():
        for control_input in op.control_inputs:
            if control_input not in control_outputs:
                control_outputs[control_input] = set()
            control_outputs[control_input].add(op)
    return control_outputs