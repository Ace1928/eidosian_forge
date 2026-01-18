from enum import Enum
from queue import Queue
from threading import Thread
from typing import Callable, Optional, List
from .errors import AsyncTaskException
def func_handler(self):
    try:
        result = self.func(*self.args, **self.kwargs)
        self.status = TaskStatus.SUCCESS
        self.result = result
        if self.on_success is not None and (not self.execute_on_caller):
            self.on_success(self.result)
    except Exception as e:
        self.exception = e
        self.status = TaskStatus.FAILURE
        if self.on_error is not None and (not self.execute_on_caller):
            self.on_error(self.exception)
    global_task_manager.remove(self.thread)