from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
def add_to_result(item: Key) -> None:
    nonlocal crit_path_counter
    next_items = [item]
    nonlocal i
    while next_items:
        item = next_items.pop()
        runnable_hull.discard(item)
        reachable_hull.discard(item)
        leaf_nodes.discard(item)
        if item in result:
            continue
        while requires_data_task[item]:
            add_to_result(requires_data_task[item].pop())
        if return_stats:
            result[item] = Order(i, crit_path_counter - _crit_path_counter_offset)
        else:
            result[item] = i
        i += 1
        for dep in dependents.get(item, ()):
            num_needed[dep] -= 1
            reachable_hull.add(dep)
            if not num_needed[dep]:
                if len(dependents[item]) == 1:
                    next_items.append(dep)
                else:
                    runnable.append(dep)