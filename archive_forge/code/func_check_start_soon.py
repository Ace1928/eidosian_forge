from typing import Awaitable, Callable
from trio import TASK_STATUS_IGNORED, Nursery, TaskStatus
def check_start_soon(nursery: Nursery) -> None:
    """start_soon() functionality."""
    nursery.start_soon(task_0)
    nursery.start_soon(task_1a)
    nursery.start_soon(task_2b)
    nursery.start_soon(task_0, 45)
    nursery.start_soon(task_1a, 32)
    nursery.start_soon(task_1b, 32)
    nursery.start_soon(task_1a, 'abc')
    nursery.start_soon(task_1b, 'abc')
    nursery.start_soon(task_2b, 'abc')
    nursery.start_soon(task_2a, 38, '46')
    nursery.start_soon(task_2c, 'abc', 12, True)
    nursery.start_soon(task_2c, 'abc', 12)
    task_2c_cast: Callable[[str, int], Awaitable[object]] = task_2c
    nursery.start_soon(task_2c_cast, 'abc', 12)
    nursery.start_soon(task_requires_kw, 12, True)
    nursery.start_soon(task_startable_1, 'cdf')