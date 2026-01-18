import importlib.util
import os
import warnings
from functools import wraps
from typing import Optional
def fail_with_message(message):
    """Generate decorator to give users message about missing TorchAudio extension."""

    def decorator(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            raise RuntimeError(f'{func.__module__}.{func.__name__} {message}')
        return wrapped
    return decorator