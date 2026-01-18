import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
@classmethod
def from_callable(cls, obj: Callable[..., Any], *, follow_wrapped: bool=True) -> 'FunctionType':
    """Generate FunctionType from a python Callable."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    parameters = [Parameter(p.name, p.kind, p.default is not p.empty, None) for p in signature.parameters.values()]
    return FunctionType(parameters)