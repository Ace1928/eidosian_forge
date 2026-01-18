import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from absl import logging
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import core
from tensorflow.python.types import trace
def sanitize_arg_name(name: str) -> str:
    """Sanitizes function argument names.

  Matches Python symbol naming rules.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
    swapped = ''.join([c if c.isalnum() else '_' for c in name])
    result = swapped if swapped[0].isalpha() else 'arg_' + swapped
    global sanitization_warnings_given
    if name != result and sanitization_warnings_given < MAX_SANITIZATION_WARNINGS:
        logging.warning('`%s` is not a valid tf.function parameter name. Sanitizing to `%s`.', name, result)
        sanitization_warnings_given += 1
    return result