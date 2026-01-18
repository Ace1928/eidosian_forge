from __future__ import annotations
import copyreg
import multiprocessing
import multiprocessing.pool
import os
import pickle
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from warnings import warn
import cloudpickle
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async, reraise
from dask.optimization import cull, fuse
from dask.system import CPU_COUNT
from dask.typing import Key
from dask.utils import ensure_dict
def initialize_worker_process(user_initializer=None):
    """
    Initialize a worker process before running any tasks in it.
    """
    default_initializer()
    if user_initializer is not None:
        user_initializer()