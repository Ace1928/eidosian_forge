import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec, overload
@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int=4, **kwargs: Any) -> None:
    """Emit debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)