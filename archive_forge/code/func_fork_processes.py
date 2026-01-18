import asyncio
import os
import multiprocessing
import signal
import subprocess
import sys
import time
from binascii import hexlify
from tornado.concurrent import (
from tornado import ioloop
from tornado.iostream import PipeIOStream
from tornado.log import gen_log
import typing
from typing import Optional, Any, Callable
def fork_processes(num_processes: Optional[int], max_restarts: Optional[int]=None) -> int:
    """Starts multiple worker processes.

    If ``num_processes`` is None or <= 0, we detect the number of cores
    available on this machine and fork that number of child
    processes. If ``num_processes`` is given and > 0, we fork that
    specific number of sub-processes.

    Since we use processes and not threads, there is no shared memory
    between any server code.

    Note that multiple processes are not compatible with the autoreload
    module (or the ``autoreload=True`` option to `tornado.web.Application`
    which defaults to True when ``debug=True``).
    When using multiple processes, no IOLoops can be created or
    referenced until after the call to ``fork_processes``.

    In each child process, ``fork_processes`` returns its *task id*, a
    number between 0 and ``num_processes``.  Processes that exit
    abnormally (due to a signal or non-zero exit status) are restarted
    with the same id (up to ``max_restarts`` times).  In the parent
    process, ``fork_processes`` calls ``sys.exit(0)`` after all child
    processes have exited normally.

    max_restarts defaults to 100.

    Availability: Unix
    """
    if sys.platform == 'win32':
        raise Exception('fork not available on windows')
    if max_restarts is None:
        max_restarts = 100
    global _task_id
    assert _task_id is None
    if num_processes is None or num_processes <= 0:
        num_processes = cpu_count()
    gen_log.info('Starting %d processes', num_processes)
    children = {}

    def start_child(i: int) -> Optional[int]:
        pid = os.fork()
        if pid == 0:
            _reseed_random()
            global _task_id
            _task_id = i
            return i
        else:
            children[pid] = i
            return None
    for i in range(num_processes):
        id = start_child(i)
        if id is not None:
            return id
    num_restarts = 0
    while children:
        pid, status = os.wait()
        if pid not in children:
            continue
        id = children.pop(pid)
        if os.WIFSIGNALED(status):
            gen_log.warning('child %d (pid %d) killed by signal %d, restarting', id, pid, os.WTERMSIG(status))
        elif os.WEXITSTATUS(status) != 0:
            gen_log.warning('child %d (pid %d) exited with status %d, restarting', id, pid, os.WEXITSTATUS(status))
        else:
            gen_log.info('child %d (pid %d) exited normally', id, pid)
            continue
        num_restarts += 1
        if num_restarts > max_restarts:
            raise RuntimeError('Too many child restarts, giving up')
        new_id = start_child(id)
        if new_id is not None:
            return new_id
    sys.exit(0)