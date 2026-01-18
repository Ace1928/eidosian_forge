from __future__ import annotations
import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator, TextIO, cast
from .application import get_app_session, run_in_terminal
from .output import Output
@property
def original_stdout(self) -> TextIO:
    return self._output.stdout or sys.__stdout__