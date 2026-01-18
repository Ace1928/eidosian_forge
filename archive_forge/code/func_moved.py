from typing import Any, Callable, Iterable, Optional, Set, TypeVar, Union
import warnings
import functools
from decorator import decorator
import numpy as np
from numpy.typing import DTypeLike
def moved(*, moved_from: str, version: str, version_removed: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark functions as moved/renamed.

    Using the decorated (old) function will result in a warning.
    """

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        warnings.warn("{:s}\n\tThis function was moved to '{:s}.{:s}' in librosa version {:s}.\n\tThis alias will be removed in librosa version {:s}.".format(moved_from, func.__module__, func.__name__, version, version_removed), category=FutureWarning, stacklevel=3)
        return func(*args, **kwargs)
    return decorator(__wrapper)