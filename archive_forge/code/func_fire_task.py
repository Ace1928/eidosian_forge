import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def fire_task():
    """Fire off a task to the thread pool"""
    key = state['ready'].pop()
    state['running'].add(key)
    for f in pretask_cbs:
        f(key, dsk, state)
    data = {dep: state['cache'][dep] for dep in get_dependencies(dsk, key)}
    apply_async(execute_task, args=(key, dumps((dsk[key], data)), dumps, loads, get_id, pack_exception), callback=queue.put)