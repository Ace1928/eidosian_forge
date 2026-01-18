from __future__ import annotations
import numbers
import sys
from contextlib import contextmanager
from functools import wraps
from importlib import metadata as importlib_metadata
from io import UnsupportedOperation
from kombu.exceptions import reraise
def coro(gen):
    """Decorator to mark generator as co-routine."""

    @wraps(gen)
    def wind_up(*args, **kwargs):
        it = gen(*args, **kwargs)
        next(it)
        return it
    return wind_up