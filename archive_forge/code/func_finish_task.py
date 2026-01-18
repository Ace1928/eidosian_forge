import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def finish_task(dsk, key, state, results, sortkey, delete=True, release_data=release_data):
    """
    Update execution state after a task finishes
    Mutates.  This should run atomically (with a lock).
    """
    for dep in sorted(state['dependents'][key], key=sortkey, reverse=True):
        s = state['waiting'][dep]
        s.remove(key)
        if not s:
            del state['waiting'][dep]
            state['ready'].append(dep)
    for dep in state['dependencies'][key]:
        if dep in state['waiting_data']:
            s = state['waiting_data'][dep]
            s.remove(key)
            if not s and dep not in results:
                if DEBUG:
                    from chest.core import nbytes
                    print('Key: %s\tDep: %s\t NBytes: %.2f\t Release' % (key, dep, sum(map(nbytes, state['cache'].values()) / 1000000.0)))
                release_data(dep, state, delete=delete)
        elif delete and dep not in results:
            release_data(dep, state, delete=delete)
    state['finished'].add(key)
    state['running'].remove(key)
    return state