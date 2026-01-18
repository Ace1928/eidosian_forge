from __future__ import annotations
import asyncio
import codecs
import itertools
import logging
import os
import select
import signal
import warnings
from collections import deque
from concurrent import futures
from typing import TYPE_CHECKING, Any, Coroutine
from tornado.ioloop import IOLoop
def pty_read(self, fd: int, events: Any=None) -> None:
    """Called by the event loop when there is pty data ready to read."""
    if not _poll(fd, timeout=0.1):
        self.log.debug('Spurious pty_read() on fd %s', fd)
        return
    ptywclients = self.ptys_by_fd[fd]
    try:
        self.pre_pty_read_hook(ptywclients)
        s = ptywclients.ptyproc.read(65536)
        ptywclients.read_buffer.append(s)
        for client in ptywclients.clients:
            client.on_pty_read(s)
    except EOFError:
        self.on_eof(ptywclients)
        for client in ptywclients.clients:
            client.on_pty_died()