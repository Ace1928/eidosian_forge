from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_heap_usage() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[float, float]]:
    """
    Returns the JavaScript heap usage.
    It is the total usage of the corresponding isolate not scoped to a particular Runtime.

    **EXPERIMENTAL**

    :returns: A tuple with the following items:

        0. **usedSize** - Used heap size in bytes.
        1. **totalSize** - Allocated heap size in bytes.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.getHeapUsage'}
    json = (yield cmd_dict)
    return (float(json['usedSize']), float(json['totalSize']))