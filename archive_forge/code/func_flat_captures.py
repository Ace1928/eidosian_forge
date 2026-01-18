import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
@property
def flat_captures(self) -> List[trace.TraceType]:
    """Flat tensor captures needed by this FunctionType."""
    if not hasattr(self, '_cached_flat_captures'):
        cached_flat_captures = []
        for t in self.captures.values():
            cached_flat_captures.extend(t._flatten())
        self._cached_flat_captures = cached_flat_captures
    return self._cached_flat_captures