from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def path_pop() -> Key:
    item = cpath_pop()
    scpath_discard(item)
    return item