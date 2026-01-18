from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Literal, TypeVar, cast, overload
from dask.typing import Graph, Key, NoDefault, no_default
def isdag(d, keys):
    """Does Dask form a directed acyclic graph when calculating keys?

    ``keys`` may be a single key or list of keys.

    Examples
    --------

    >>> inc = lambda x: x + 1
    >>> inc = lambda x: x + 1
    >>> isdag({'x': 0, 'y': (inc, 'x')}, 'y')
    True
    >>> isdag({'x': (inc, 'y'), 'y': (inc, 'x')}, 'y')
    False

    See Also
    --------
    getcycle
    """
    return not getcycle(d, keys)