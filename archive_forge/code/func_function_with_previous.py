import os
from typing import Any, TypeVar, Callable, Optional, cast
from typing import Protocol
def function_with_previous(func: F) -> _FunctionWithPrevious[F]:
    """Decorate a function as having an attribute named 'previous'."""
    function_with_previous = cast(_FunctionWithPrevious[F], func)
    function_with_previous.previous = None
    return function_with_previous