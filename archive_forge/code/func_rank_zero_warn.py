import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec, overload
@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int=4, **kwargs: Any) -> None:
    """Emit warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)