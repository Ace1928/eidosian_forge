import copy
import functools
import itertools
import multiprocessing.pool
import os
import queue
import re
import types
import warnings
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from multiprocessing import Manager
from queue import Empty
from shutil import disk_usage
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
from urllib.parse import urlparse
import multiprocess
import multiprocess.pool
import numpy as np
from tqdm.auto import tqdm
from .. import config
from ..parallel import parallel_map
from . import logging
from . import tqdm as hf_tqdm
from ._dill import (  # noqa: F401 # imported for backward compatibility. TODO: remove in 3.0.0
def iflatmap_unordered(pool: Union[multiprocessing.pool.Pool, multiprocess.pool.Pool], func: Callable[..., Iterable[Y]], *, kwargs_iterable: Iterable[dict]) -> Iterable[Y]:
    initial_pool_pid = _get_pool_pid(pool)
    pool_changed = False
    manager_cls = Manager if isinstance(pool, multiprocessing.pool.Pool) else multiprocess.Manager
    with manager_cls() as manager:
        queue = manager.Queue()
        async_results = [pool.apply_async(_write_generator_to_queue, (queue, func, kwargs)) for kwargs in kwargs_iterable]
        try:
            while True:
                try:
                    yield queue.get(timeout=0.05)
                except Empty:
                    if all((async_result.ready() for async_result in async_results)) and queue.empty():
                        break
                if _get_pool_pid(pool) != initial_pool_pid:
                    pool_changed = True
                    raise RuntimeError('One of the subprocesses has abruptly died during map operation.To debug the error, disable multiprocessing.')
        finally:
            if not pool_changed:
                [async_result.get(timeout=0.05) for async_result in async_results]