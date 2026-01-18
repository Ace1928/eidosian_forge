import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def most_specific_common_subtype(self, others: Sequence['FunctionType']) -> Optional['FunctionType']:
    """Returns a common subtype (if exists)."""
    subtyped_parameters = []
    for i, parameter in enumerate(self.parameters.values()):
        subtyped_parameter = parameter.most_specific_common_supertype([list(other.parameters.values())[i] for other in others])
        if subtyped_parameter is None:
            return None
        subtyped_parameters.append(subtyped_parameter)
    if not all(subtyped_parameters):
        return None
    capture_names = set(self.captures.keys())
    for other in others:
        capture_names = capture_names.union(other.captures.keys())
    subtyped_captures = collections.OrderedDict()
    for name in capture_names:
        containing = [t for t in [self, *others] if name in t.captures]
        base = containing[0]
        relevant_others = containing[1:]
        common_type = base.captures[name].most_specific_common_supertype([other.captures[name] for other in relevant_others])
        if common_type is None:
            return None
        else:
            subtyped_captures[name] = common_type
    return FunctionType(subtyped_parameters, subtyped_captures)