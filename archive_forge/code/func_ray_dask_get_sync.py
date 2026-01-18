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
def ray_dask_get_sync(dsk, keys, **kwargs):
    """
    A synchronous Dask-Ray scheduler. This scheduler will send top-level
    (non-inlined) Dask tasks to a Ray cluster for execution. The scheduler will
    wait for the tasks to finish executing, fetch the results, and repackage
    them into the appropriate Dask collections. This particular scheduler
    submits Ray tasks synchronously, which can be useful for debugging.

    This can be passed directly to `dask.compute()`, as the scheduler:

    >>> dask.compute(obj, scheduler=ray_dask_get_sync)

    You can override the currently active global Dask-Ray callbacks (e.g.
    supplied via a context manager):

    >>> dask.compute(
            obj,
            scheduler=ray_dask_get_sync,
            ray_callbacks=some_ray_dask_callbacks,
        )

    Args:
        dsk: Dask graph, represented as a task DAG dictionary.
        keys (List[str]): List of Dask graph keys whose values we wish to
            compute and return.

    Returns:
        Computed values corresponding to the provided keys.
    """
    ray_callbacks = kwargs.pop('ray_callbacks', None)
    persist = kwargs.pop('ray_persist', False)
    with local_ray_callbacks(ray_callbacks) as ray_callbacks:
        ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs, ray_postsubmit_all_cbs, ray_finish_cbs = unpack_ray_callbacks(ray_callbacks)
        object_refs = get_async(_apply_async_wrapper(apply_sync, _rayify_task_wrapper, ray_presubmit_cbs, ray_postsubmit_cbs, ray_pretask_cbs, ray_posttask_cbs), 1, dsk, keys, **kwargs)
        if ray_postsubmit_all_cbs is not None:
            for cb in ray_postsubmit_all_cbs:
                cb(object_refs, dsk)
        del dsk
        if persist:
            result = object_refs
        else:
            result = ray_get_unpack(object_refs)
        if ray_finish_cbs is not None:
            for cb in ray_finish_cbs:
                cb(result)
        return result