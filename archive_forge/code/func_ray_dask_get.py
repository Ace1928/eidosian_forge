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
def ray_dask_get(dsk, keys, **kwargs):
    """
    A Dask-Ray scheduler. This scheduler will send top-level (non-inlined) Dask
    tasks to a Ray cluster for execution. The scheduler will wait for the
    tasks to finish executing, fetch the results, and repackage them into the
    appropriate Dask collections. This particular scheduler uses a threadpool
    to submit Ray tasks.

    This can be passed directly to `dask.compute()`, as the scheduler:

    >>> dask.compute(obj, scheduler=ray_dask_get)

    You can override the currently active global Dask-Ray callbacks (e.g.
    supplied via a context manager), the number of threads to use when
    submitting the Ray tasks, or the threadpool used to submit Ray tasks:

    >>> dask.compute(
            obj,
            scheduler=ray_dask_get,
            ray_callbacks=some_ray_dask_callbacks,
            num_workers=8,
            pool=some_cool_pool,
        )

    Args:
        dsk: Dask graph, represented as a task DAG dictionary.
        keys (List[str]): List of Dask graph keys whose values we wish to
            compute and return.
        ray_callbacks (Optional[list[callable]]): Dask-Ray callbacks.
        num_workers (Optional[int]): The number of worker threads to use in
            the Ray task submission traversal of the Dask graph.
        pool (Optional[ThreadPool]): A multiprocessing threadpool to use to
            submit Ray tasks.

    Returns:
        Computed values corresponding to the provided keys.
    """
    num_workers = kwargs.pop('num_workers', None)
    pool = kwargs.pop('pool', None)
    global default_pool
    thread = threading.current_thread()
    if pool is None:
        with pools_lock:
            if num_workers is None and thread is main_thread:
                if default_pool is None:
                    default_pool = ThreadPool(CPU_COUNT)
                    atexit.register(default_pool.close)
                pool = default_pool
            elif thread in pools and num_workers in pools[thread]:
                pool = pools[thread][num_workers]
            else:
                pool = ThreadPool(num_workers)
                atexit.register(pool.close)
                pools[thread][num_workers] = pool
    ray_callbacks = kwargs.pop('ray_callbacks', None)
    persist = kwargs.pop('ray_persist', False)
    enable_progress_bar = kwargs.pop('_ray_enable_progress_bar', None)
    if 'resources' in kwargs:
        raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)
    ray_remote_args = kwargs.pop('ray_remote_args', {})
    try:
        annotations = dask.config.get('annotations')
    except KeyError:
        annotations = {}
    if 'resources' in annotations:
        raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)
    scoped_ray_remote_args = _build_key_scoped_ray_remote_args(dsk, annotations, ray_remote_args)
    with local_ray_callbacks(ray_callbacks) as ray_callbacks:
        ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs, ray_postsubmit_all_cbs, ray_finish_cbs = unpack_ray_callbacks(ray_callbacks)
        object_refs = get_async(_apply_async_wrapper(pool.apply_async, _rayify_task_wrapper, ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs, scoped_ray_remote_args), len(pool._pool), dsk, keys, get_id=_thread_get_id, pack_exception=pack_exception, **kwargs)
        if ray_postsubmit_all_cbs is not None:
            for cb in ray_postsubmit_all_cbs:
                cb(object_refs, dsk)
        del dsk
        if persist:
            result = object_refs
        else:
            pb_actor = None
            if enable_progress_bar:
                pb_actor = ray.get_actor('_dask_on_ray_pb')
            result = ray_get_unpack(object_refs, progress_bar_actor=pb_actor)
        if ray_finish_cbs is not None:
            for cb in ray_finish_cbs:
                cb(result)
    with pools_lock:
        active_threads = set(threading.enumerate())
        if thread is not main_thread:
            for t in list(pools):
                if t not in active_threads:
                    for p in pools.pop(t).values():
                        p.close()
    return result