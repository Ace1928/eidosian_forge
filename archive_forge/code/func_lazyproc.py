from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
def lazyproc(name: str='lazyproc', start: bool=True, daemon: bool=False):

    def wrapper_proc(func):

        def wrapped_proc(*args, **kwargs):
            proc = LazyProc(name, *args, start=start, daemon=daemon, **kwargs)
            proc = LazyProcs.add_proc(proc)
            return proc
        return wrapped_proc
    return wrapper_proc