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
@isolation_level.setter
def isolation_level(self, value: IsolationLevel) -> None:
    self._conn.isolation_level = value