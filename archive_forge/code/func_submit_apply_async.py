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
def submit_apply_async(apply_async, fn, *args, **kwargs):
    fut = Future()
    apply_async(fn, args, kwargs, fut.set_result, fut.set_exception)
    return fut