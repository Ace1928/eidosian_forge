import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def placeholder_arguments(self, placeholder_context: trace.PlaceholderContext) -> inspect.BoundArguments:
    """Returns BoundArguments of values that can be used for tracing."""
    arguments = collections.OrderedDict()
    for parameter in self.parameters.values():
        if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
            raise ValueError('Can not generate placeholder values for variable length function type.')
        if not parameter.type_constraint:
            raise ValueError('Can not generate placeholder value for partially defined function type.')
        placeholder_context.update_naming_scope(parameter.name)
        arguments[parameter.name] = parameter.type_constraint.placeholder_value(placeholder_context)
    return inspect.BoundArguments(self, arguments)