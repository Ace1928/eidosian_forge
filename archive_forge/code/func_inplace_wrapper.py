from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging
def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        val = fn(gm)
        return gm
    return wrapped_fn