from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
@classmethod
def kill_proc(cls, name: str):
    if name in LazyProcs.active_procs:
        proc = LazyProcs.active_procs.pop(name)
        proc.terminate()
        LazyProcs.inactive_procs[name] = proc
        LazyProcs.set_state()