from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def stop_worker_processes(cls, name: str, verbose: bool=True, timeout: float=5.0, kind: Optional[str]=None):
    """
        Stops the worker processes
        """
    if kind is None:
        kind = 'default'
    if cls.worker_processes.get(kind) is None or cls.worker_processes[kind].get(name) is None:
        if verbose:
            cls.logger.warning(f'[{kind.capitalize()}] No worker processes found for {name}')
        return
    curr_proc, n_procs = (0, len(cls.worker_processes[kind][name]))
    while cls.worker_processes[kind][name]:
        proc = cls.worker_processes[kind][name].pop()
        if proc._closed:
            continue
        log_name = f'`|g|{name}-{curr_proc}|e|`'
        if verbose:
            cls.logger.info(f'- [|y|{kind.capitalize():7s}|e|] Stopping: [ {curr_proc + 1}/{n_procs} ] {log_name:20s} (Process ID: {proc.pid})', colored=True)
        proc.join(timeout)
        proc.terminate()
        try:
            proc.close()
        except Exception as e:
            if verbose:
                cls.logger.info(f'|y|[{kind.capitalize():7s}]|e| Failed Stop: [ {curr_proc + 1}/{n_procs} ] {log_name:20s} (Error: |r|{e}|e|)', colored=True)
            try:
                signal.pthread_kill(proc.ident, signal.SIGKILL)
                proc.join(timeout)
                proc.terminate()
            except Exception as e:
                if verbose:
                    cls.logger.info(f'|r|[{kind.capitalize():7s}]|e| Failed Kill: [ {curr_proc + 1}/{n_procs} ] {log_name:20s} (Error: |r|{e}|e|)', colored=True)
            with contextlib.suppress(Exception):
                proc.kill()
                proc.close()
        curr_proc += 1