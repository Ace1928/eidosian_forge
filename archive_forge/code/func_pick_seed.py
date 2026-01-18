from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def pick_seed() -> Key | None:
    while occurences_grouped_sorted:
        key = max(occurences_grouped_sorted)
        picked_root = occurences_grouped_sorted[key][-1]
        if picked_root in result:
            occurences_grouped_sorted[key].pop()
            if not occurences_grouped_sorted[key]:
                del occurences_grouped_sorted[key]
            continue
        return picked_root
    return None