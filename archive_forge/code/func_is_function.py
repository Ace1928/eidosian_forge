import itertools
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import resource_variable_ops
def is_function(fname, graph):
    """Checks for a function definition with `fname` in the current context."""
    if context.executing_eagerly():
        return context.context().has_function(fname)
    else:
        while graph is not None:
            if graph._is_function(fname):
                return True
            if hasattr(graph, 'outer_graph'):
                graph = graph.outer_graph
            else:
                return False