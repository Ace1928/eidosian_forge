import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def pack_output(self, flat_values: Sequence[core.Tensor]) -> Any:
    """Packs flat tensors to generate a value of the output type."""
    if flat_values is None:
        flat_values = []
    if self.output is None:
        raise ValueError('Can not pack outputs for undefined output type.')
    else:
        return self.output._from_tensors(iter(flat_values))