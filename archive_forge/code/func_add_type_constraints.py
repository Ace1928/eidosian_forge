import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def add_type_constraints(function_type: FunctionType, input_signature: Any, default_values: Dict[str, Any]) -> FunctionType:
    """Adds type constraints to a FunctionType based on the input_signature."""
    context = trace_type.InternalTracingContext(is_legacy_signature=True)
    constraints = [trace_type.from_value(c, context) for c in input_signature]
    parameters = []
    has_var_pos = any((p.kind is p.VAR_POSITIONAL for p in function_type.parameters.values()))
    for param in function_type.parameters.values():
        sanitized_kind = param.POSITIONAL_ONLY if has_var_pos and param.kind is param.POSITIONAL_OR_KEYWORD else param.kind
        if param.name == 'self':
            parameters.append(Parameter('self', sanitized_kind, param.optional, None))
        elif param.kind is param.VAR_KEYWORD:
            continue
        elif param.kind is param.VAR_POSITIONAL:
            for i in range(len(constraints)):
                parameters.append(Parameter(param.name + '_' + str(i), Parameter.POSITIONAL_ONLY, False, constraints.pop(0)))
        elif param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY]:
            if param.kind is param.KEYWORD_ONLY and param.name not in default_values:
                raise TypeError(f'Since input_signature is defined, keyword-only parameter `{param.name}` must have a default value')
            if constraints:
                parameters.append(Parameter(param.name, sanitized_kind, param.optional, constraints.pop(0)))
            elif param.name in default_values:
                type_constraint = trace_type.from_value(default_values[param.name])
                parameters.append(Parameter(param.name, sanitized_kind, param.optional, type_constraint))
            else:
                raise TypeError(f'input_signature missing type constraint for {param.name}')
    if constraints:
        raise TypeError(f'input_signature contains {len(constraints)} extra type constraints.')
    return FunctionType(parameters)