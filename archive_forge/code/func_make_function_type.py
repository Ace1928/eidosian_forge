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
def make_function_type(python_function, input_signature):
    """Generates a FunctionType for python_function."""
    _validate_signature(input_signature)
    function_type = function_type_lib.FunctionType.from_callable(python_function)
    default_values = function_type_lib.FunctionType.get_default_values(python_function)
    if input_signature is not None:
        input_signature = tuple(input_signature)
        function_type = function_type_lib.add_type_constraints(function_type, input_signature, default_values)
    return (function_type, default_values)