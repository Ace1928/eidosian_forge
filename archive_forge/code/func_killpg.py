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
def killpg(self, sig: int=signal.SIGTERM) -> Any:
    """Send a signal to the process group of the process in the pty"""
    if os.name == 'nt':
        return self.ptyproc.kill(sig)
    pgid = os.getpgid(self.ptyproc.pid)
    os.killpg(pgid, sig)
    return None