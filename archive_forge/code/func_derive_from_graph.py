import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def derive_from_graph(func_graph):
    """Derives a FunctionType from FuncGraph."""
    input_signature = (tuple((trace_type.from_value(i) for i in func_graph.inputs)), {})
    output_signature = tuple((trace_type.from_value(o) for o in func_graph.outputs))
    return function_type_lib.from_structured_signature(input_signature, output_signature, func_graph.function_captures.capture_types)