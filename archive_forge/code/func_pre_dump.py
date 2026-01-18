from __future__ import annotations
import functools
from typing import Any, Callable, cast
def pre_dump(fn: Callable[..., Any] | None=None, pass_many: bool=False) -> Callable[..., Any]:
    """Register a method to invoke before serializing an object. The method
    receives the object to be serialized and returns the processed object.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema`'s :func:`~marshmallow.Schema.dump` call.
    If ``pass_many=True``, the raw data (which may be a collection) is passed.

    .. versionchanged:: 3.0.0
        ``many`` is always passed as a keyword arguments to the decorated method.
    """
    return set_hook(fn, (PRE_DUMP, pass_many))