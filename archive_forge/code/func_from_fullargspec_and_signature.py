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
@classmethod
def from_fullargspec_and_signature(cls, fullargspec, input_signature, is_pure=False, name=None, jit_compile=None):
    """Construct FunctionSpec from legacy FullArgSpec format."""
    function_type, default_values = to_function_type(fullargspec)
    if input_signature:
        input_signature = tuple(input_signature)
        _validate_signature(input_signature)
        function_type = function_type_lib.add_type_constraints(function_type, input_signature, default_values)
    return FunctionSpec(function_type, default_values, is_pure, name, jit_compile)