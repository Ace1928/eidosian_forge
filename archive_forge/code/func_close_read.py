from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import ContextManager, Iterator, TextIO, cast
from ..utils import DummyContext
from .base import PipeInput
from .vt100 import Vt100Input
def close_read(self) -> None:
    """Close read-end if not yet closed."""
    if self._read_closed:
        return
    os.close(self.read_fd)
    self._read_closed = True