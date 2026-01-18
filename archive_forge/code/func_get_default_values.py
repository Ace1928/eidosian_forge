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
def get_default_values(cls, obj: Callable[..., Any], *, follow_wrapped: bool=True) -> Dict[str, Any]:
    """Inspects and returns a dictionary of default values."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    default_values = {}
    for p in signature.parameters.values():
        if p.default is not p.empty:
            default_values[p.name] = p.default
    return default_values