from __future__ import annotations
from collections.abc import Callable
from contextlib import contextmanager
from typing import ClassVar
def normalize_callback(cb):
    """Normalizes a callback to a tuple"""
    if isinstance(cb, Callback):
        return cb._callback
    elif isinstance(cb, tuple):
        return cb
    else:
        raise TypeError('Callbacks must be either `Callback` or `tuple`')