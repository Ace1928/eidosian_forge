import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def from_structured_signature(input_signature=None, output_signature=None, capture_types=None) -> FunctionType:
    """Generates a FunctionType from legacy signature representation."""
    if input_signature is None:
        input_signature = ((), {})
    args, kwargs = input_signature
    parameters = []
    for i, arg in enumerate(args):
        parameters.append(Parameter('arg_' + str(i), Parameter.POSITIONAL_ONLY, False, trace_type.from_value(arg, trace_type.InternalTracingContext(is_legacy_signature=True))))
    for name, kwarg in kwargs.items():
        parameters.append(Parameter(sanitize_arg_name(name), Parameter.KEYWORD_ONLY, False, trace_type.from_value(kwarg, trace_type.InternalTracingContext(is_legacy_signature=True))))
    return_type = trace_type.from_value(output_signature, trace_type.InternalTracingContext(is_legacy_signature=True))
    return FunctionType(parameters, capture_types or {}, return_annotation=return_type)