from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
def run_module_with_profiler(module, profiler, backend, passed_args=[]):
    from runpy import run_module
    builtins.__dict__['profile'] = profiler
    ns = dict(_CLEAN_GLOBALS, profile=profiler)
    _backend = choose_backend(backend)
    sys.argv = [module] + passed_args
    if _backend == 'tracemalloc' and has_tracemalloc:
        tracemalloc.start()
    try:
        run_module(module, run_name='__main__', init_globals=ns)
    finally:
        if has_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()