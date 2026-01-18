from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def add_server_process(cls, name: str, cmd: str, verbose: bool=True):
    """
        Adds the server process
        """
    if verbose:
        cls.logger.info(f'Adding Server Process: {cmd}', prefix=name)
    context = multiprocessing.get_context('spawn')
    p = context.Process(target=os.system, args=(cmd,))
    p.start()
    cls.server_processes.append(p)
    return p