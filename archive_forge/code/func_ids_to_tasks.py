from .task_list import task_list
from collections import defaultdict
def ids_to_tasks(ids):
    if ids is None:
        raise RuntimeError('No task specified. Please select a task with ' + '--task {task_name}.')
    return ','.join((_id_to_task(i) for i in ids.split(',') if len(i) > 0))