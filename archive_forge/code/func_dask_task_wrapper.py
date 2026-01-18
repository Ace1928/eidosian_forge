import atexit
import threading
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Optional
import ray
import dask
from dask.core import istask, ishashable, _execute_task
from dask.system import CPU_COUNT
from dask.threaded import pack_exception, _thread_get_id
from ray.util.dask.callbacks import local_ray_callbacks, unpack_ray_callbacks
from ray.util.dask.common import unpack_object_refs
from ray.util.dask.scheduler_utils import get_async, apply_sync
@ray.remote
def dask_task_wrapper(func, repack, key, ray_pretask_cbs, ray_posttask_cbs, *args):
    """
    A Ray remote function acting as a Dask task wrapper. This function will
    repackage the given flat `args` into its original data structures using
    `repack`, execute any Dask subtasks within the repackaged arguments
    (inlined by Dask's optimization pass), and then pass the concrete task
    arguments to the provide Dask task function, `func`.

    Args:
        func: The Dask task function to execute.
        repack: A function that repackages the provided args into
            the original (possibly nested) Python objects.
        key: The Dask key for this task.
        ray_pretask_cbs: Pre-task execution callbacks.
        ray_posttask_cbs: Post-task execution callback.
        *args (ObjectRef): Ray object references representing the Dask task's
            arguments.

    Returns:
        The output of the Dask task. In the context of Ray, a
        dask_task_wrapper.remote() invocation will return a Ray object
        reference representing the Ray task's result.
    """
    if ray_pretask_cbs is not None:
        pre_states = [cb(key, args) if cb is not None else None for cb in ray_pretask_cbs]
    repacked_args, repacked_deps = repack(args)
    actual_args = [_execute_task(a, repacked_deps) for a in repacked_args]
    result = func(*actual_args)
    if ray_posttask_cbs is not None:
        for cb, pre_state in zip(ray_posttask_cbs, pre_states):
            if cb is not None:
                cb(key, result, pre_state)
    return result