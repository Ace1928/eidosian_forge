from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
def cached_property(func: typing.Callable) -> property:
    cached_name = f'_cached_{func}'
    sentinel = object()

    def inner(instance: object):
        cache = getattr(instance, cached_name, sentinel)
        if cache is not sentinel:
            return cache
        result = func(instance)
        setattr(instance, cached_name, result)
        return result
    return property(inner)