from .errors import AsyncTaskException
from .types import (
from .utils import (
from typing import Callable
def set_max_threads(num: int):
    global_task_manager.max_threads(num)