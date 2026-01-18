import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec, overload
def rank_zero_deprecation(message: Union[str, Warning], stacklevel: int=5, **kwargs: Any) -> None:
    """Emit a deprecation warning only on global rank 0."""
    category = kwargs.pop('category', rank_zero_deprecation_category)
    rank_zero_warn(message, stacklevel=stacklevel, category=category, **kwargs)