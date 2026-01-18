from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def start_server_process(cls, cmd: str, verbose: bool=True):
    """
        Starts the server process
        """
    if verbose:
        cls.logger.info(f'Starting Server Process: {cmd}')
    context = multiprocessing.get_context('spawn')
    p = context.Process(target=os.system, args=(cmd,))
    p.start()
    cls.state.add_leader_process_id(p.pid, 'server')
    cls.server_processes.append(p)
    return p