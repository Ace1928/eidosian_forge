from __future__ import annotations
import copy
from collections import defaultdict, deque, namedtuple
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Callable, Literal, NamedTuple, overload
from dask.core import get_dependencies, get_deps, getcycle, istask, reverse_dict
from dask.typing import Key
@_with_offset
def process_runnables() -> None:
    """Compute all currently runnable paths and either cache or execute them

        This is designed to ensure we are running tasks that are free to execute
        (e.g. the result of a splitter task) not too eagerly. If we executed
        such free tasks too early we'd be walking the graph in a too wide /
        breadth first fashion that is not optimal. If instead we were to only
        execute them once they are needed for a final result, this can cause
        very high memory pressure since valuable reducers are executed too
        late.

        The strategy here is to take all runnable tasks and walk forwards until
        we hit a reducer node (i.e. a node with more than one dependency). We
        will remember/cache the path to this reducer node.
        If this path leads to a leaf or if we find enough runnable paths for a
        reducer to be runnable, we will execute the path.

        If instead of a reducer a splitter is encountered that is runnable, we
        will follow its splitter paths individually and apply the same logic to
        each branch.
        """
    while runnable:
        candidates = runnable.copy()
        runnable.clear()
        while candidates:
            key = candidates.pop()
            if key in runnable_hull or key in result:
                continue
            if key in leaf_nodes:
                add_to_result(key)
                continue
            path = [key]
            branches = deque([path])
            while branches:
                path = branches.popleft()
                while True:
                    current = path[-1]
                    runnable_hull.add(current)
                    deps_downstream = dependents[current]
                    deps_upstream = dependencies[current]
                    if not deps_downstream:
                        if num_needed[current] <= 1:
                            for k in path:
                                add_to_result(k)
                    elif len(path) == 1 or len(deps_upstream) == 1:
                        if len(deps_downstream) > 1:
                            for d in sorted(deps_downstream, key=sort_key):
                                if len(dependencies[d]) == 1:
                                    branch = path.copy()
                                    branch.append(d)
                                    branches.append(branch)
                            break
                        runnable_hull.update(deps_downstream)
                        path.extend(sorted(deps_downstream, key=sort_key))
                        continue
                    elif current in known_runnable_paths:
                        known_runnable_paths[current].append(path)
                        if len(known_runnable_paths[current]) >= num_needed[current]:
                            pruned_branches: deque[list[Key]] = deque()
                            for path in known_runnable_paths_pop(current):
                                if path[-2] not in result:
                                    pruned_branches.append(path)
                            if len(pruned_branches) < num_needed[current]:
                                known_runnable_paths[current] = list(pruned_branches)
                            else:
                                while pruned_branches:
                                    path = pruned_branches.popleft()
                                    for k in path:
                                        if num_needed[k]:
                                            pruned_branches.append(path)
                                            break
                                        add_to_result(k)
                    elif len(dependencies[current]) > 1 and num_needed[current] <= 1:
                        for k in path:
                            add_to_result(k)
                    else:
                        known_runnable_paths[current] = [path]
                    break