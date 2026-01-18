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
def node_from_map(node_map, name):
    """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]