from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
@classmethod
def killall(cls):
    for name in LazyProcs.active_procs:
        LazyProcs.kill_proc(name)