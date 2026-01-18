from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def start_worker_processes(cls, name: str, kind: Optional[str]=None, config: Optional[Dict[str, Any]]=None, num_workers: Optional[int]=1, verbose: Optional[bool]=True, base_index: Optional[Union[int, str]]='auto', spawn_worker_func: Optional[Union[Callable, str]]=None, num_worker_func: Optional[Union[Callable, str]]=None, get_worker_func: Optional[Union[Callable, str]]=None, settings_config_func: Optional[Union[Callable, str]]=None, **kwargs):
    """
        Starts the worker processes
        """
    procs = []
    if kind is None:
        kind = 'default'
    if verbose:
        cls.logger.info(f'[{kind.capitalize()}] Spawning Worker: {name} ({num_workers})')
    context = multiprocessing.get_context('spawn')
    if base_index == 'auto':
        base_index = get_base_worker_index()
    if num_worker_func is None:
        num_worker_func = cls.get_settings_func(cls.get_num_worker_func)
    if get_worker_func is None:
        get_worker_func = cls.get_settings_func(cls.get_worker_func)
    if settings_config_func is None:
        settings_config_func = cls.get_settings_func(cls.settings_config_func)
    if spawn_worker_func is None:
        from .workers import spawn_new_worker
        spawn_worker_func = spawn_new_worker
    if num_worker_func:
        num_workers = num_worker_func(name=name, num_workers=num_workers, kind=kind)
    if not kwargs:
        kwargs = {}
    kwargs['kind'] = kind
    kwargs['config'] = config
    kwargs['verbose'] = verbose
    kwargs['settings_config_func'] = settings_config_func
    kwargs['get_worker_func'] = get_worker_func
    for n in range(num_workers):
        is_primary_worker = n == 0
        worker_index = base_index * num_workers + n
        kwargs['is_primary_worker'] = is_primary_worker
        kwargs['index'] = worker_index
        p = context.Process(target=spawn_worker_func, args=(name,), kwargs=kwargs)
        p.start()
        if verbose:
            log_name = f'{name}-{worker_index}'
            cls.logger.info(f'- |g|[{kind.capitalize()}]|e| Started: [ {n + 1}/{num_workers} ] `|g|{log_name:20s}|e|` (Process ID: {p.pid})', colored=True)
        procs.append(p)
    cls.add_worker_processes(kind=kind, name=name, procs=procs)
    return procs