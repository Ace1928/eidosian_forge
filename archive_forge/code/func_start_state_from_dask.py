import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def start_state_from_dask(dsk, cache=None, sortkey=None):
    """Start state from a dask
    Examples
    --------
    >>> dsk = {
        'x': 1,
        'y': 2,
        'z': (inc, 'x'),
        'w': (add, 'z', 'y')}  # doctest: +SKIP
    >>> from pprint import pprint  # doctest: +SKIP
    >>> pprint(start_state_from_dask(dsk))  # doctest: +SKIP
    {'cache': {'x': 1, 'y': 2},
     'dependencies': {'w': {'z', 'y'}, 'x': set(), 'y': set(), 'z': {'x'}},
     'dependents': {'w': set(), 'x': {'z'}, 'y': {'w'}, 'z': {'w'}},
     'finished': set(),
     'ready': ['z'],
     'released': set(),
     'running': set(),
     'waiting': {'w': {'z'}},
     'waiting_data': {'x': {'z'}, 'y': {'w'}, 'z': {'w'}}}
    """
    if sortkey is None:
        sortkey = order(dsk).get
    if cache is None:
        cache = config.get('cache', None)
    if cache is None:
        cache = dict()
    data_keys = set()
    for k, v in dsk.items():
        if not has_tasks(dsk, v):
            cache[k] = v
            data_keys.add(k)
    dsk2 = dsk.copy()
    dsk2.update(cache)
    dependencies = {k: get_dependencies(dsk2, k) for k in dsk}
    waiting = {k: v.copy() for k, v in dependencies.items() if k not in data_keys}
    dependents = reverse_dict(dependencies)
    for a in cache:
        for b in dependents.get(a, ()):
            waiting[b].remove(a)
    waiting_data = {k: v.copy() for k, v in dependents.items() if v}
    ready_set = {k for k, v in waiting.items() if not v}
    ready = sorted(ready_set, key=sortkey, reverse=True)
    waiting = {k: v for k, v in waiting.items() if v}
    state = {'dependencies': dependencies, 'dependents': dependents, 'waiting': waiting, 'waiting_data': waiting_data, 'cache': cache, 'ready': ready, 'running': set(), 'finished': set(), 'released': set()}
    return state