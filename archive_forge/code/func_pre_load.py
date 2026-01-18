from __future__ import annotations
import functools
from typing import Any, Callable, cast
def pre_load(fn: Callable[..., Any] | None=None, pass_many: bool=False) -> Callable[..., Any]:
    """Register a method to invoke before deserializing an object. The method
    receives the data to be deserialized and returns the processed data.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema`'s :func:`~marshmallow.Schema.load` call.
    If ``pass_many=True``, the raw data (which may be a collection) is passed.

    .. versionchanged:: 3.0.0
        ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    """
    return set_hook(fn, (PRE_LOAD, pass_many))