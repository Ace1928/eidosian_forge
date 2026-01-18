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
def from_function_and_signature(cls, python_function, input_signature, is_pure=False, jit_compile=None):
    """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    function_type, default_values = make_function_type(python_function, input_signature)
    while isinstance(python_function, functools.partial):
        python_function = python_function.func
    name = getattr(python_function, '__name__', 'f')
    return FunctionSpec(function_type, default_values, is_pure=is_pure, jit_compile=jit_compile, name=name)