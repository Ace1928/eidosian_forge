from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
def register_keypaths(cls: type[CustomTreeNode[T]], handler: KeyPathHandler) -> KeyPathHandler:
    """Register a key path handler for a custom pytree node type."""
    if not inspect.isclass(cls):
        raise TypeError(f'Expected a class, got {cls}.')
    if cls in _KEYPATH_REGISTRY:
        raise ValueError(f'Key path handler for {cls} has already been registered.')
    _KEYPATH_REGISTRY[cls] = handler
    return handler