from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def get_child_process_ids(cls, name: Optional[str]=None, kind: Optional[str]=None, process_id: Optional[int]=None, first: Optional[bool]=True) -> List[int]:
    """
        Returns the child process ids
        """
    if not _psutil_available:
        cls.logger.warning('psutil not available. Cannot get child process ids')
        return []
    if not name and (not process_id):
        raise ValueError('Must provide either name or process_id')
    if process_id:
        proc = psutil.Process(process_id)
        return [child.pid for child in proc.children(recursive=True)]
    if name in {'server', 'app'} and kind is None:
        procs = []
        for proc in cls.server_processes:
            if proc._closed:
                continue
            parent = psutil.Process(proc.pid)
            procs.extend((child.pid for child in parent.children(recursive=True)))
            if first:
                break
        return procs
    if kind is None:
        kind = 'default'
    if kind not in cls.worker_processes:
        cls.logger.warning(f'No worker processes found for {kind}')
        return []
    if name not in cls.worker_processes[kind]:
        cls.logger.warning(f'No worker processes found for {kind}.{name}')
        return []
    procs = []
    for proc in cls.worker_processes[kind][name]:
        if proc._closed:
            continue
        parent = psutil.Process(proc.pid)
        procs.extend((child.pid for child in parent.children(recursive=True)))
        if first:
            break
    return procs