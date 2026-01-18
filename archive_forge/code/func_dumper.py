import asyncio
import logging
import sqlite3
from functools import partial
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Thread
from typing import (
from warnings import warn
from .context import contextmanager
from .cursor import Cursor
def dumper():
    try:
        for line in self._conn.iterdump():
            dump_queue.put_nowait(line)
        dump_queue.put_nowait(None)
    except Exception:
        LOG.exception('exception while dumping db')
        dump_queue.put_nowait(None)
        raise