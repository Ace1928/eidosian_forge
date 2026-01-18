import os
from queue import Queue, Empty
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
def execute_task(key, task_info, dumps, loads, get_id, pack_exception):
    """
    Compute task and handle all administration
    See Also
    --------
    _execute_task : actually execute task
    """
    try:
        task, data = loads(task_info)
        result = _execute_task(task, data)
        id = get_id()
        result = dumps((result, id))
        failed = False
    except BaseException as e:
        result = pack_exception(e, dumps)
        failed = True
    return (key, result, failed)