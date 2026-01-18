from __future__ import annotations
import contextlib
import re
import sqlite3
from typing import cast, Any, Iterable, Iterator, Tuple
from coverage.debug import auto_repr, clipped_repr, exc_one_line
from coverage.exceptions import DataError
from coverage.types import TDebugCtl
def executescript(self, script: str) -> None:
    """Same as :meth:`python:sqlite3.Connection.executescript`."""
    if self.debug.should('sql'):
        self.debug.write('Executing script with {} chars: {}'.format(len(script), clipped_repr(script, 100)))
    assert self.con is not None
    self.con.executescript(script).close()