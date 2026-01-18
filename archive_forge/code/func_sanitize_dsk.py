from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def sanitize_dsk(dsk: MutableMapping[Key, Any]) -> dict:
    """Take a dask graph and replace callables with a dummy function and remove
    payload data like numpy arrays, dataframes, etc.
    """
    new = {}
    for key, values in dsk.items():
        new_key = key
        new[new_key] = _convert_task(values)
    if get_deps(new) != get_deps(dsk):
        raise RuntimeError('Sanitization failed to preserve topology.')
    return new