from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload
from dask.typing import Graph, Key, NoDefault, no_default
def istask(x):
    """Is x a runnable task?

    A task is a tuple with a callable first argument

    Examples
    --------

    >>> inc = lambda x: x + 1
    >>> istask((inc, 1))
    True
    >>> istask(1)
    False
    """
    return type(x) is tuple and x and callable(x[0])