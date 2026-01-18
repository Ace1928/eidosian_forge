import collections
import pprint
import re
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import func_graph as func_graph_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def set_preinitialized_function_spec(concrete_fn, spec):
    """Set the FunctionType of the ConcreteFunction using FunctionSpec."""
    if spec is None:
        concrete_fn._function_type = None
        return
    unconstrained_type = function_type_lib.FunctionType([function_type_lib.Parameter(p.name, p.kind, p.optional, None) for p in spec.function_type.parameters.values()])
    arg_specs, kwarg_specs = concrete_fn.structured_input_signature
    input_function_type, _ = function_type_lib.canonicalize_to_monomorphic(arg_specs, {function_type_lib.sanitize_arg_name(k): v for k, v in kwarg_specs.items()}, spec.default_values, {}, unconstrained_type)
    output_type = trace_type.from_value(concrete_fn.graph.structured_outputs)
    function_type = function_type_lib.FunctionType(input_function_type.parameters.values(), return_annotation=output_type)
    concrete_fn._function_type = function_type