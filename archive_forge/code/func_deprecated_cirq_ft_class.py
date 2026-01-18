import functools
import unittest.mock
import os
from typing import Callable, Type
from cirq._compat import deprecated, deprecated_class
def deprecated_cirq_ft_class() -> Callable[[Type], Type]:
    """Decorator to mark a class in Cirq-FT deprecated."""
    return deprecated_class(deadline=_DEPRECATION_DEADLINE, fix=_DEPRECATION_FIX_MSG)