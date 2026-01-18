from typing import List
from .types import AsyncTask, TaskStatus, WaitType, ErrorHandling
from .errors import AsyncTaskException
def await_first(tasks: List[AsyncTask], error_handling: ErrorHandling=ErrorHandling.RAISE):
    """
    Await first task to complete

    :param tasks: List of tasks
    :param error_handling: Error handling strategy (raise or ignore)
    """
    first_failed: AsyncTask = None
    while tasks:
        for i in range(len(tasks)):
            if tasks[i].status == TaskStatus.SUCCESS:
                return tasks[i].result
            elif tasks[i].status == TaskStatus.FAILURE:
                if first_failed is None:
                    first_failed = tasks[i]
                tasks.pop(i)
                i -= 1
    if first_failed is not None and error_handling == ErrorHandling.RAISE:
        raise AsyncTaskException(str(first_failed.exception), first_failed.func.__name__)
    return None