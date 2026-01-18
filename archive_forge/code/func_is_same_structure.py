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
def is_same_structure(structure1, structure2, check_values=False):
    """Check two structures for equality, optionally of types and of values."""
    try:
        nest.assert_same_structure(structure1, structure2, expand_composites=True)
    except (ValueError, TypeError):
        return False
    if check_values:
        flattened1 = nest.flatten(structure1, expand_composites=True)
        flattened2 = nest.flatten(structure2, expand_composites=True)
        if any((type(f1) is not type(f2) for f1, f2 in zip(flattened1, flattened2))):
            return False
        return flattened1 == flattened2
    return True