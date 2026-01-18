import re
import sys
import types
import unittest
from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
def assertion(graph):
    matches = []
    for node in graph.as_graph_def().node:
        if re.match(op_regex, node.name):
            matches.append(node)
    for fn in graph.as_graph_def().library.function:
        for node_def in fn.node_def:
            if re.match(op_regex, node_def.name):
                matches.append(node_def)
    self.assertLen(matches, n)