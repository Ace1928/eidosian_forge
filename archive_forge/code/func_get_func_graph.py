from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib
def get_func_graph(op, input_shapes, func_name):
    """Generates and returns a FuncGraph for the given op and input_shapes."""
    fdef = None
    graph = op.graph
    while graph is not None:
        func = graph._get_function(func_name)
        if func is not None:
            fdef = func.cached_definition
            break
        if hasattr(graph, 'outer_graph'):
            graph = graph.outer_graph
        else:
            break
    if fdef is None:
        raise KeyError('%s cannot be found in the graph' % func_name)
    with op.graph.as_default():
        func_graph = function_def_to_graph.function_def_to_graph(fdef, input_shapes=input_shapes)
    for operation in func_graph.get_operations():
        if operation.type in ['PartitionedCall', 'StatefulPartitionedCall']:
            f = graph._get_function(operation.get_attr('f').name)
            try:
                cf = concrete_function.ConcreteFunction.from_func_graph(f.graph, f.function_type, attrs=f.cached_definition.attr)
            except AttributeError:
                continue
            operation._gradient_function = cf._get_gradient_function()
    return func_graph