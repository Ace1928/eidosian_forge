from __future__ import annotations
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future
from functools import partial
from queue import Empty, Queue
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
from dask.typing import Key
def batch_execute_tasks(it):
    """
    Batch computing of multiple tasks with `execute_task`
    """
    return [execute_task(*a) for a in it]