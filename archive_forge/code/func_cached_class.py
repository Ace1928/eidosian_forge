from __future__ import annotations
import os
from functools import wraps
from typing import Hashable, TypeVar
def cached_class(klass: type[Klass]) -> type[Klass]:
    """
    Decorator to cache class instances by constructor arguments.
    This results in a class that behaves like a singleton for each
    set of constructor arguments, ensuring efficiency.

    Note that this should be used for *immutable classes only*.  Having
    a cached mutable class makes very little sense.  For efficiency,
    avoid using this decorator for situations where there are many
    constructor arguments permutations.

    The keywords argument dictionary is converted to a tuple because
    dicts are mutable; keywords themselves are strings and
    so are always hashable, but if any arguments (keyword
    or positional) are non-hashable, that set of arguments
    is not cached.
    """
    cache: dict[tuple[Hashable, ...], type[Klass]] = {}

    @wraps(klass, assigned=('__name__', '__module__'), updated=())
    class _decorated(klass):
        __doc__ = klass.__doc__

        def __new__(cls, *args, **kwargs):
            """
            Pass through.
            """
            key = (cls, *args, *tuple(kwargs.items()))
            try:
                inst = cache.get(key)
            except TypeError:
                inst = key = None
            if inst is None:
                inst = klass(*args, **kwargs)
                inst.__class__ = cls
                if key is not None:
                    cache[key] = inst
            return inst

        def __init__(self, *args, **kwargs):
            pass
    return _decorated