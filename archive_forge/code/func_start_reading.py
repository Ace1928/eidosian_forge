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
def start_reading(self, ptywclients: PtyWithClients) -> None:
    """Connect a terminal to the tornado event loop to read data from it."""
    fd = ptywclients.ptyproc.fd
    self.ptys_by_fd[fd] = ptywclients
    loop = IOLoop.current()
    loop.add_handler(fd, self.pty_read, loop.READ)