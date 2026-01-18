from __future__ import annotations
import multiprocessing
import os
import sys
from multiprocessing.context import SpawnProcess
from socket import socket
from typing import Callable
from uvicorn.config import Config
def get_subprocess(config: Config, target: Callable[..., None], sockets: list[socket]) -> SpawnProcess:
    """
    Called in the parent process, to instantiate a new child process instance.
    The child is not yet started at this point.

    * config - The Uvicorn configuration instance.
    * target - A callable that accepts a list of sockets. In practice this will
               be the `Server.run()` method.
    * sockets - A list of sockets to pass to the server. Sockets are bound once
                by the parent process, and then passed to the child processes.
    """
    try:
        stdin_fileno = sys.stdin.fileno()
    except (AttributeError, OSError):
        stdin_fileno = None
    kwargs = {'config': config, 'target': target, 'sockets': sockets, 'stdin_fileno': stdin_fileno}
    return spawn.Process(target=subprocess_started, kwargs=kwargs)