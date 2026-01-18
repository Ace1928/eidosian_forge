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
def get_op_and_outputs(op_or_outputs):
    if isinstance(op_or_outputs, ops.Operation):
        return (op_or_outputs, [])
    elif not op_or_outputs:
        return (None, [])
    else:
        return (op_or_outputs[0].op, op_or_outputs)